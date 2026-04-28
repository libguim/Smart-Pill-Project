"""
pills/db_fallback.py

ResNet 인식 실패 시 각인 + 색상 + 모양을 DB와 매핑해
후보 약품을 찾아주는 보완 검색 모듈.

사용 위치: views.py — predicted_idx is None 블록
"""

import re
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


# ── 색상 감지 (필루미 벤치마킹: KMeans 클러스터링) ────────────────
# 기존 HSV 범위 비교는 배경색이 섞여 오감지가 잦았음.
# 필루미 방식: KMeans(k=3)로 픽셀을 클러스터링 후
# 2번째로 큰 비율의 색 = 알약 색 (1등은 대부분 배경)으로 판단.

def _hsv_to_color_name(h: float, s: float, v: float) -> str:
    """
    HSV 단일 값을 DB color_class1 문자열로 변환.
    필루미 Configuration.GetColor 로직을 한국어 DB 컬럼에 맞게 적용.
    """
    if s < 20 and v > 150:          return "하양"
    if v < 70:                       return "검정"
    if h <= 170 and s < 50:         return "분홍"   # 저채도 고색조(분홍) → 빨강보다 먼저 체크
    if h < 10 or h > 160:           return "빨강"
    if h <= 22:                      return "주황"
    if h <= 38:                      return "노랑"
    if h <= 85:                      return "초록"
    if h <= 130:                     return "파랑"
    if h <= 155:                     return "보라"
    if h <= 170:                     return "분홍"
    if s < 40:                       return "갈색"
    return "기타"


def detect_color(pil_image: Image.Image) -> str:
    """
    KMeans 클러스터링으로 알약의 주요 색상을 감지합니다. (필루미 벤치마킹)

    k=3으로 픽셀을 3개 그룹으로 분류 후:
      - 비율 1위: 대부분 배경
      - 비율 2위: 알약 본체 색상 ← 채택
      - 비율 3위: 각인·그림자 등 소수

    단일 픽셀 HSV 비교 대비 배경·조명 영향을 크게 줄입니다.
    """
    img_np = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 전체 픽셀을 1D로 펼쳐서 KMeans 입력
    reshape = hsv.reshape((-1, 3)).astype(float)

    try:
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=0).fit(reshape)
        labels = np.arange(0, len(np.unique(kmeans.labels_)) + 1)
        hist, _ = np.histogram(kmeans.labels_, bins=labels)
        hist = hist.astype(float) / hist.sum()

        # 비율 오름차순 정렬 → [-2]가 2번째로 큰 클러스터 = 알약 색
        colors = sorted(zip(hist, kmeans.cluster_centers_))
        h, s, v = colors[-2][1]

        color_name = _hsv_to_color_name(h, s, v)
        print(f"[COLOR] KMeans 감지: {color_name} (H:{h:.0f} S:{s:.0f} V:{v:.0f})")
        return color_name

    except Exception as e:
        # KMeans 실패 시 중앙 픽셀 단순 판단으로 폴백
        print(f"[COLOR] KMeans 실패({e}) → 중앙 픽셀 폴백")
        cy, cx = hsv.shape[0] // 2, hsv.shape[1] // 2
        h, s, v = hsv[cy, cx].astype(float)
        return _hsv_to_color_name(h, s, v)


# ── 모양 감지 (필루미 벤치마킹: 배경 픽셀 제거 후 윤곽선 분석) ──────

def detect_shape(pil_image: Image.Image) -> str:
    """
    알약 모양을 감지합니다. (필루미 ShapePreprocess 벤치마킹)

    필루미는 배경을 검정(0)으로 처리한 뒤 임계값 5로 이진화해
    노이즈 없이 알약 실루엣만 추출합니다.
    우리는 color_pil(원본 크롭)을 받으므로 Otsu + 윤곽선으로 동일 효과 구현.

    반환: '원형' | '타원형' | '장방형' | '반원형' | '기타'
    """
    img_np = np.array(pil_image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 필루미 방식: 배경이 어두운 경우 단순 이진화로 충분
    # Otsu로 최적 임계값 자동 결정
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 작은 노이즈 제거 (필루미 make_fit_size 역할)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "기타"

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 500:
        return "기타"

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h

    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return "기타"

    # 원형도: 1.0에 가까울수록 완전한 원
    circularity = 4 * np.pi * area / (perimeter ** 2)

    # 볼록도: 실제 면적 / 볼록 껍질 면적 (오목한 형태 감지용)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    if circularity > 0.82:
        shape = "원형"
    elif 4 <= len(cv2.approxPolyDP(cnt, 0.04 * perimeter, True)) <= 6 and solidity > 0.85:
        shape = "사각형"
    elif circularity > 0.65 and aspect_ratio > 1.25:
        shape = "타원형"
    elif aspect_ratio > 1.8 and solidity > 0.85:
        shape = "장방형"
    elif solidity < 0.75:
        shape = "반원형"
    else:
        shape = "원형"

    vertices = len(cv2.approxPolyDP(cnt, 0.04 * perimeter, True))
    print(f"[SHAPE] 감지: {shape} (원형도:{circularity:.2f}, 꼭짓점:{vertices}, 비율:{aspect_ratio:.2f}, 볼록도:{solidity:.2f})")
    return shape


# ── OCR 텍스트 정제 ───────────────────────────────────

def clean_ocr_text(raw_results: list) -> str:
    """EasyOCR/PaddleOCR 결과 리스트를 정제해 각인 매핑용 텍스트 반환."""
    raw = "".join(raw_results).upper().replace(" ", "")
    cleaned = re.sub(r'[^A-Z0-9/]', '', raw)
    if len(cleaned) > 12 or len(cleaned) < 1:
        return ""
    return cleaned


def _ocr_from_versions(ocr_img, reader) -> str:
    """
    ocr_img가 리스트(다중 전처리 버전)이든 단일 ndarray이든
    EasyOCR로 모두 시도해 최빈값을 채택합니다.
    동률이면 가장 짧은 결과를 채택합니다.
    """
    from collections import Counter

    versions = ocr_img if isinstance(ocr_img, list) else [ocr_img]
    results_list = []

    for i, img in enumerate(versions):
        try:
            results = reader.readtext(img, detail=0)
            text = clean_ocr_text(results)
            print(f"[FALLBACK_OCR] v{i+1}: '{text}'")
            if text:
                results_list.append(text)
        except Exception as e:
            print(f"[FALLBACK_OCR] v{i+1} 오류: {e}")

    # ── 최빈값 채택 전략 ─────────────────────────────
    if results_list:
        count = Counter(results_list)
        top_result, top_count = count.most_common(1)[0]
        if top_count >= 2:
            best = top_result  # 2회 이상 동일 → 신뢰도 높음
        else:
            best = min(results_list, key=len)  # 모두 다르면 가장 짧은 것
    else:
        best = ""
        try:
            from paddleocr import PaddleOCR
            paddle = PaddleOCR(use_angle_cls=True, lang='en',
                               show_log=False, use_gpu=False)
            for i, img in enumerate(versions[:2]):
                img_rgb = (cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                           if len(img.shape) == 2 else img)
                result = paddle.ocr(img_rgb, cls=True)
                if result and result[0]:
                    texts = [line[1][0] for line in result[0] if line[1][1] > 0.5]
                    text = clean_ocr_text(texts)
                    print(f"[FALLBACK_PADDLE] v{i+1}: '{text}'")
                    if text:
                        best = text
                        break
        except Exception as e:
            print(f"[FALLBACK_PADDLE] 실패: {e}")

    return best


# ── DB 보완 검색 ──────────────────────────────────────

def db_fallback_search(
    pil_image: Image.Image,
    ocr_img,        # list[ndarray] 또는 단일 ndarray
    reader,         # easyocr.Reader (전역 인스턴스 재사용)
    PillMaster,     # Django 모델
) -> dict | None:
    """
    ResNet 실패 시 각인 + 색상 + 모양으로 DB 보완 검색.

    우선순위:
      1순위: 각인 완전 일치
      2순위: 각인 + 색상 일치
      3순위: 각인 + 모양 일치
      4순위: 색상 + 모양 일치 (후보 5개 이하일 때만)

    반환: {'pill': PillMaster 객체, 'method': str, 'candidates': int}
          못 찾으면 None
    """

    # ── Step 1: OCR 각인 추출 ────────────────────────
    ocr_text = ""
    if ocr_img is not None:
        ocr_text = _ocr_from_versions(ocr_img, reader)
        print(f"[FALLBACK_OCR] 최종 각인 텍스트: '{ocr_text}'")

    # ── Step 2: 색상·모양 감지 ───────────────────────
    detected_color = detect_color(pil_image)
    detected_shape = detect_shape(pil_image)

    # ── Step 3: 각인 완전 일치 + 색상 검증 ──────────
    # 각인+색상 동시 조건으로 오인식 각인 방어
    # 예) OCR이 GSEE1→88 오인식 시, 씬지록신정(분홍)과 라믹탈정 사진(노랑) 색상이 달라 걸러짐
    if ocr_text:
        pill = (
            PillMaster.objects.filter(
                print_front__iexact=ocr_text,
                color_class1__icontains=detected_color
            ).first()
            or PillMaster.objects.filter(
                print_back__iexact=ocr_text,
                color_class1__icontains=detected_color
            ).first()
        )
        if pill:
            print(f"✅ [FALLBACK] 각인+색상 완전일치 → {pill.dl_name}")
            return {"pill": pill, "method": "각인 일치", "candidates": 1}

    # ── Step 3.5: 각인 완전일치 (색상 무관) ──────────
    # 조건: 색상 불일치 + DB 유일 후보 + OCR이 DB 각인과 완전히 동일한 경우
    # 비모보정: OCR='500/20', DB 각인='500/20' → 완전 동일 → 색상 오감지여도 안전하게 반환
    # 라믹탈정: OCR='88', DB 각인='88' → 동일하지만 실제로는 오인식
    #           → 단, 씬지록신정 색상=분홍, 감지 색상=노랑 → Step 3에서 이미 걸러짐
    #           → Step 3.5까지 오지 않음 (Step 3에서 불일치 후 여기로 오면 이미 색상 검증 통과 못한 것)
    # 따라서 색상 불일치여도 OCR=DB각인이면 반환하되,
    # 단 감지된 색상과 DB 색상이 완전히 반대(예: 노랑vs분홍)인 경우는 제외
    if ocr_text:
        front_qs = PillMaster.objects.filter(print_front__iexact=ocr_text)
        back_qs  = PillMaster.objects.filter(print_back__iexact=ocr_text)
        all_qs   = (front_qs | back_qs).distinct()
        count    = all_qs.count()
        if count == 1:
            candidate = all_qs.first()
            db_color = (candidate.color_class1 or "").strip()
            # DB 색상과 감지 색상이 완전히 다른 유채색이면 오인식으로 판단 → 건너뜀
            # 예) 감지=노랑, DB=분홍 → 오인식 가능성 높음
            # 예) 감지=하양, DB=노랑 → 색상 감지 오류 가능성 (연한 노랑 → 하양 오감지)
            chromatic = {"빨강","주황","노랑","초록","파랑","보라","분홍","갈색"}
            both_chromatic = detected_color in chromatic and db_color in chromatic
            color_conflict = both_chromatic and detected_color != db_color

            if not color_conflict:
                print(f"✅ [FALLBACK] 각인 유일 후보 + 색상 허용 → {candidate.dl_name}")
                return {"pill": candidate, "method": "각인 일치", "candidates": 1}
            else:
                print(f"[FALLBACK] 각인 유일 후보이나 색상 충돌({detected_color}≠{db_color}) → 건너뜀")

    # ── Step 4: 각인 + 색상 일치 ─────────────────────
    if ocr_text:
        qs = (
            PillMaster.objects.filter(
                print_front__icontains=ocr_text,
                color_class1__icontains=detected_color
            ) | PillMaster.objects.filter(
                print_back__icontains=ocr_text,
                color_class1__icontains=detected_color
            )
        )
        if qs.exists():
            pill = qs.first()
            print(f"✅ [FALLBACK] 각인+색상 일치 → {pill.dl_name}")
            return {"pill": pill, "method": "각인+색상 일치", "candidates": qs.count()}

    # ── Step 5: 각인 + 모양 일치 ─────────────────────
    if ocr_text:
        qs = (
            PillMaster.objects.filter(
                print_front__icontains=ocr_text,
                drug_shape__icontains=detected_shape
            ) | PillMaster.objects.filter(
                print_back__icontains=ocr_text,
                drug_shape__icontains=detected_shape
            )
        )
        if qs.exists():
            pill = qs.first()
            print(f"✅ [FALLBACK] 각인+모양 일치 → {pill.dl_name}")
            return {"pill": pill, "method": "각인+모양 일치", "candidates": qs.count()}

    # ── Step 6: 색상 + 모양 일치 ─────────────────────
    # 후보가 너무 많으면 의미 없으므로 5개 이하일 때만 반환
    if detected_color != "기타":
        qs = PillMaster.objects.filter(
            color_class1__icontains=detected_color,
            drug_shape__icontains=detected_shape
        )
        count = qs.count()
        if 1 <= count <= 5:
            pill = qs.first()
            print(f"✅ [FALLBACK] 색상+모양 일치 ({count}개 후보) → {pill.dl_name}")
            return {"pill": pill, "method": "색상+모양 일치", "candidates": count}
        else:
            print(f"[FALLBACK] 색상+모양 후보 {count}개 → 너무 많아 반환 안 함")

    print(f"❌ [FALLBACK] 매칭 실패 (각인:'{ocr_text}', 색상:{detected_color}, 모양:{detected_shape})")
    return None