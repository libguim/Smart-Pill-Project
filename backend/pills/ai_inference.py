import re
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import easyocr
import cv2
from PIL import Image
from torchvision import models, transforms
from transformers import CLIPProcessor, CLIPModel
from django.conf import settings

# ── 디바이스 및 모델 경로 ──────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 1000
model_path = os.path.join(settings.BASE_DIR, 'pills', 'ai_models', 'pill_resnet152_dataclass01_aug0.pt')

# ── ResNet-152 로드 ────────────────────────────────────
model = models.resnet152(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    print("✅ AI 모델 로드 완료")
model.to(device).eval()

# ── CLIP 및 OCR 전역 로드 ──────────────────────────────
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
reader = easyocr.Reader(['en'])

# ── CNN용 전처리 ───────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── 임계값 상수 ───────────────────────────────────────
CONF_THRESHOLD        = 0.30   # ResNet 최소 신뢰도
CONF_HIGH             = 0.65   # 이 이상이면 ResNet "확신" 상태
ENERGY_THRESHOLD      = 8.0    # OOD 감지용 에너지 임계값
ENTROPY_THRESHOLD     = 1.0    # Top-5 엔트로피 상한 (높을수록 혼란)
GAP_THRESHOLD         = 0.08   # Top1-Top2 점수 차이 하한 (작을수록 불확실)
CLIP_VERIFY_THRESHOLD = 0.45   # CLIP 교차검증 최소 점수
OCR_MAX_LEN           = 10     # 이 이상의 텍스트는 규격표/워터마크로 판단


def _compute_entropy(scores: list) -> float:
    """Top-5 확률분포의 엔트로피 (scipy 없이)"""
    return -sum(p * math.log(p + 1e-9) for p in scores)


# 유사 문자 정규화 테이블 (OCR이 자주 혼동하는 글자 쌍)
# MB→NB, 0→O, 1→I 등의 오인식 대응
SIMILAR_CHARS = {
    'N': 'M', 'M': 'N',   # M↔N
    '0': 'O', 'O': '0',   # 0↔O
    '1': 'I', 'I': '1',   # 1↔I
    '8': 'B', 'B': '8',   # 8↔B
    '5': 'S', 'S': '5',   # 5↔S
    '6': 'G', 'G': '6',   # 6↔G
}

def _normalize_ocr(text: str) -> str:
    """OCR 유사 문자를 정규화해서 비교 정확도를 높입니다."""
    return ''.join(SIMILAR_CHARS.get(c, c) for c in text.upper())


def _is_resnet_confident(top5_conf, top5_preds):
    """
    ResNet 예측의 확신 여부 판단.
    반환: (확신여부: bool, 예측 idx: int, 신뢰도: float)
    """
    scores = top5_conf[0].cpu().numpy().tolist()
    top1_score = scores[0]
    gap = top1_score - scores[1]
    entropy = _compute_entropy(scores)
    predicted_idx = int(top5_preds[0][0].item())

    is_confident = (
        top1_score >= CONF_THRESHOLD
        and gap >= GAP_THRESHOLD
        and entropy <= ENTROPY_THRESHOLD
    )
    print(f"[RESNET] Idx:{predicted_idx} Top1:{top1_score:.4f} Gap:{gap:.4f} Entropy:{entropy:.4f} → {'확신' if is_confident else '불확실'}")
    return is_confident, predicted_idx, top1_score


def _run_ocr(ocr_img) -> str:
    """
    다중 전처리 버전(리스트)을 받아 EasyOCR로 모두 시도하고
    최빈값을 채택합니다. 동률이면 가장 짧은 결과를 채택합니다.
    (기존 "가장 긴 결과" 전략은 노이즈 문자열을 선택하는 문제가 있었음)
    """
    from collections import Counter

    if isinstance(ocr_img, np.ndarray):
        versions = [ocr_img]
    else:
        versions = ocr_img

    results_list = []

    for i, img in enumerate(versions):
        try:
            results = reader.readtext(img, detail=0)
            raw = re.sub(r'[^A-Z0-9/\-]', '', "".join(results).upper())

            if len(raw) > OCR_MAX_LEN:
                print(f"[OCR] v{i+1} 과다({len(raw)}자) 무시")
                continue

            print(f"[OCR] v{i+1}: '{raw}'")
            if raw:
                results_list.append(raw)
        except Exception as e:
            print(f"[OCR] v{i+1} 오류: {e}")
            continue

    # ── 최빈값 채택 전략 ─────────────────────────────
    if results_list:
        count = Counter(results_list)
        top_result, top_count = count.most_common(1)[0]
        if top_count >= 2:
            best = top_result  # 2회 이상 동일 → 신뢰도 높음
        else:
            best = min(results_list, key=len)  # 모두 다르면 가장 짧은 것
        print(f"[OCR] 최종 채택: '{best}' (후보: {results_list})")
    else:
        best = ""
        try:
            from paddleocr import PaddleOCR
            paddle = PaddleOCR(use_angle_cls=True, lang='en',
                               show_log=False, use_gpu=False)
            for i, img in enumerate(versions[:2]):
                if len(img.shape) == 2:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    img_rgb = img
                result = paddle.ocr(img_rgb, cls=True)
                if result and result[0]:
                    texts = [line[1][0] for line in result[0] if line[1][1] > 0.5]
                    raw = re.sub(r'[^A-Z0-9/\-]', '', "".join(texts).upper())
                    print(f"[OCR_PADDLE] v{i+1}: '{raw}'")
                    if raw and len(raw) <= OCR_MAX_LEN:
                        best = raw
                        break
        except Exception as e:
            print(f"[OCR_PADDLE] 실패: {e}")

    return best


def _verify_with_clip(image: Image.Image, pill_name: str) -> bool:
    """ResNet 예측 결과를 CLIP으로 교차검증 (시각적 특징 기반)"""
    verify_labels = [
        "a single medicine pill or tablet on a clean background",
        "a document, chart, or diagram with text and numbers",
        "multiple objects or a cluttered background",
        "a photo that is not clearly a single pill",
    ]
    inputs = clip_processor(
        text=verify_labels, images=image,
        return_tensors="pt", padding=True
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

    match_score = float(probs[0][0])
    print(f"[CLIP_VERIFY] 점수: {match_score:.4f} → {'통과' if match_score >= CLIP_VERIFY_THRESHOLD else '거부'}")
    return match_score >= CLIP_VERIFY_THRESHOLD


def predict_pill(image_obj, ocr_img: np.ndarray = None, pill_data_map: dict = None):
    """
    듀얼 파이프라인 (CNN + OCR) 추론.

    Args:
        image_obj:     CNN용 PIL Image (utils.preprocess_for_inference 반환값 첫 번째)
        ocr_img:       OCR용 대비강화 numpy 배열 (utils.preprocess_for_inference 반환값 두 번째)
        pill_data_map: 각인 DB 매핑 딕셔너리 (id_mapping)

    반환: (predicted_idx, confidence) — idx=None 이면 인식 실패
    """
    try:
        image = image_obj if isinstance(image_obj, Image.Image) else Image.open(image_obj).convert('RGB')

        # ── Step 1: CLIP 알약 여부 필터 ─────────────────
        clip_labels = [
            "a close-up photo of a medicine tablet or capsule pill",
            "a photo of food or snack",
            "a photo of a flower or plant",
            "a photo of an animal or person",
            "a photo of a landscape or object",
        ]
        clip_inputs = clip_processor(
            text=clip_labels, images=image,
            return_tensors="pt", padding=True
        ).to(device)

        with torch.no_grad():
            clip_outputs = clip_model(**clip_inputs)
            clip_probs = clip_outputs.logits_per_image.softmax(dim=1)

        if torch.argmax(clip_probs) != 0:
            pill_score = float(clip_probs[0][0])
            print(f"🚫 [CLIP_REJECT] 알약 아님 (Pill Score: {pill_score:.4f})")
            return None, pill_score

        # ── Step 2: ResNet 추론 ──────────────────────────
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            energy = torch.logsumexp(logits, dim=1).item()
            probs = F.softmax(logits, dim=1)
            top5_conf, top5_preds = torch.topk(probs, 5)

        if energy < ENERGY_THRESHOLD:
            print(f"⚠️ [OOD_REJECT] Energy {energy:.4f} < {ENERGY_THRESHOLD}")
            return None, float(top5_conf[0][0].item())

        resnet_confident, resnet_idx, resnet_conf = _is_resnet_confident(top5_conf, top5_preds)

        # ── Step 3: OCR — utils에서 생성한 전용 이미지 사용 ──
        # ocr_img가 없으면 CNN 이미지에서 직접 추출 (하위 호환)
        if ocr_img is not None:
            detected_text = _run_ocr(ocr_img)
        else:
            img_np = np.array(image)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(gray)
            detected_text = _run_ocr(enhanced)

        # ── Step 4: OCR × DB 각인 대조 ──────────────────
        ocr_matched_idx = None
        ocr_matched_conf = None

        if detected_text and pill_data_map:
            for i in range(5):
                p_idx = int(top5_preds[0][i].item())
                p_conf = float(top5_conf[0][i].item())
                info = pill_data_map.get(str(p_idx), {})
                f_print = str(info.get('print_front', '')).upper().strip()
                b_print = str(info.get('print_back', '')).upper().strip()

                front_match = bool(f_print) and (
                    detected_text == f_print
                    or f_print in detected_text
                    or _normalize_ocr(detected_text) == _normalize_ocr(f_print)  # 유사 문자 허용
                )
                back_match = bool(b_print) and (
                    detected_text == b_print
                    or b_print in detected_text
                    or _normalize_ocr(detected_text) == _normalize_ocr(b_print)  # 유사 문자 허용
                )

                if front_match or back_match:
                    match_type = "완전" if (detected_text == f_print or detected_text == b_print) else "부분"
                    print(f"🎯 [OCR_MATCH] '{detected_text}' {match_type}일치 → Idx {p_idx}, Conf {p_conf:.4f}")
                    ocr_matched_idx = p_idx
                    ocr_matched_conf = min(p_conf + 0.20, 1.0)
                    break

        # ── Step 5: 결합 판단 ────────────────────────────

        # Case 1: ResNet 확신 + OCR 일치 → 최고 신뢰 케이스
        if resnet_confident and ocr_matched_idx is not None and ocr_matched_idx == resnet_idx:
            final_conf = min(resnet_conf + 0.20, 1.0)
            print(f"✅ [DUAL_MATCH] ResNet+OCR 모두 일치 → Idx {resnet_idx}, Conf {final_conf:.4f}")
            return resnet_idx, final_conf

        # Case 2: ResNet 확신 + OCR 텍스트 감지됐으나 DB 불일치
        if resnet_confident and ocr_matched_idx is None and detected_text:
            # ResNet이 가리키는 약의 각인과 OCR 결과 사이에 공통 글자가 있으면
            # OCR 오인식 가능성 → ResNet 우선 신뢰
            # 예) 마이암부톨: OCR=NB, 매핑=YH/MB → B 겹침 → ResNet 신뢰
            # 예) 라믹탈정:  OCR=88, 매핑=ADCX   → 공통 글자 없음 → FALLBACK
            mapped_f = (pill_data_map or {}).get(str(resnet_idx), {}).get('print_front', '').upper()
            mapped_b = (pill_data_map or {}).get(str(resnet_idx), {}).get('print_back', '').upper()
            all_mapped = mapped_f + mapped_b
            overlap = set(detected_text) & set(all_mapped)

            if overlap and resnet_conf >= 0.80:
                pill_name = (pill_data_map or {}).get(str(resnet_idx), {}).get('dl_name', '')
                if _verify_with_clip(image, pill_name):
                    print(f"✅ [RESNET_HIGH_CONF] 각인 글자 겹침{overlap} → ResNet 우선 → Idx {resnet_idx}")
                    return resnet_idx, resnet_conf

            print(f"❌ [OCR_MISMATCH] OCR '{detected_text}' DB 불일치 → FALLBACK으로 넘김")
            return None, resnet_conf

        # Case 3: ResNet 확신 + OCR 텍스트 없음 → CLIP 교차검증
        if resnet_confident and ocr_matched_idx is None and not detected_text:
            pill_name = (pill_data_map or {}).get(str(resnet_idx), {}).get('dl_name', '')
            if _verify_with_clip(image, pill_name):
                print(f"✅ [RESNET+CLIP] CLIP 교차검증 통과 → Idx {resnet_idx}, Conf {resnet_conf:.4f}")
                return resnet_idx, resnet_conf
            else:
                print(f"❌ [CLIP_VERIFY_FAIL] CLIP 교차검증 실패 → 거부")
                return None, resnet_conf

        # Case 4: ResNet 불확실 + OCR 일치 → OCR 주도
        if not resnet_confident and ocr_matched_idx is not None:
            print(f"✅ [OCR_LEAD] ResNet 불확실, OCR 주도 → Idx {ocr_matched_idx}, Conf {ocr_matched_conf:.4f}")
            return ocr_matched_idx, ocr_matched_conf

        # Case 5: ResNet 확신 + OCR이 다른 인덱스와 일치 → CLIP 중재
        if resnet_confident and ocr_matched_idx is not None and ocr_matched_idx != resnet_idx:
            print(f"⚠️ [CONFLICT] ResNet({resnet_idx}) vs OCR({ocr_matched_idx}) → CLIP 중재")
            pill_name = (pill_data_map or {}).get(str(ocr_matched_idx), {}).get('dl_name', '')
            if _verify_with_clip(image, pill_name):
                print(f"✅ [OCR_WIN] CLIP이 OCR 지지 → Idx {ocr_matched_idx}")
                return ocr_matched_idx, ocr_matched_conf
            else:
                print(f"✅ [RESNET_WIN] CLIP이 OCR 거부, ResNet 유지 → Idx {resnet_idx}")
                return resnet_idx, resnet_conf

        # Case 6: 둘 다 불확실 → 재촬영 요청
        print(f"❌ [UNCERTAIN] ResNet 불확실 + OCR 미매칭 → 거부")
        return None, resnet_conf

    except Exception as e:
        print(f"❌ 추론 중 에러: {e}")
        return None, 0.0