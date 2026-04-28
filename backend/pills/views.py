import os
import json
import math
from django.conf import settings
from rest_framework import viewsets
from rest_framework.viewsets import ReadOnlyModelViewSet
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.forms.models import model_to_dict
from .models import PillMaster, PillDetail
from .serializers import PillMasterSerializer, PillDetailSerializer
from .ai_inference import predict_pill, reader
from .ai_engine.utils import preprocess_for_inference
from .db_fallback import db_fallback_search

# 매핑 데이터 로드 (서버 시작 시 1회)
MAPPING_FILE_PATH = settings.BASE_DIR / 'pills' / 'ai_models' / 'pill_index_map.json'
try:
    with open(MAPPING_FILE_PATH, 'r', encoding='utf-8') as f:
        id_mapping = json.load(f)
    print(f"✅ 매핑 데이터 로드 완료: {len(id_mapping)} 개 항목")
except Exception as e:
    print(f"❌ 매핑 파일 로드 실패: {e}")
    id_mapping = {}


def _confidence_label(pct: float) -> str:
    """신뢰도 수치 → TTS 친화적 표현 (소수점 2자리, 100% 방지)"""
    label = "매우 높음" if pct >= 90 else "높음" if pct >= 70 else "보통" if pct >= 50 else "낮음"
    display = min(round(pct, 2), 99.99)
    return f"{label}, {display:.2f}퍼센트"


def _method_label(method: str) -> str:
    """인식 방법명 → TTS 친화적 한국어 (기술 용어 병행)"""
    return {
        "ResNet":           "CNN 딥러닝 이미지 인식, ResNet-152 모델 사용",
        "각인 일치":         "OCR 문자 인식",
        "각인+색상 일치":    "OCR 문자 인식 및 색상 분석",
        "각인+모양 일치":    "OCR 문자 인식 및 모양 분석",
        "색상+모양 일치":    "색상 및 모양 분석, Color & Shape 조합",
        "색상+모양 참고":    "색상 및 모양 참고 분석",
    }.get(method, method)


def _confidence_for_fallback(method: str) -> str:
    """fallback 방법에 따른 confidence 문장 (TTS 자연스럽게)"""
    return {
        "각인 일치":         "OCR 문자 인식으로 확인했습니다",
        "각인+색상 일치":    "OCR 문자 인식과 색상 분석으로 확인했습니다",
        "각인+모양 일치":    "OCR 문자 인식과 모양 분석으로 확인했습니다",
        "색상+모양 일치":    "색상과 모양 분석으로 확인했습니다",
        "색상+모양 참고":    "색상과 모양을 참고하여 확인했습니다",
    }.get(method, "보조 분석으로 확인했습니다")


def _build_pill_response(pill_master, pill_detail, extra: dict) -> dict:
    """고정된 키 순서로 응답을 조립합니다."""
    m = model_to_dict(pill_master) if pill_master else {}
    d = model_to_dict(pill_detail) if pill_detail else {}

    master_created_at = (
        pill_master.created_at.strftime('%Y-%m-%d %H:%M:%S')
        if pill_master and pill_master.created_at else None
    )
    crawled_at = (
        pill_detail.crawled_at.strftime('%Y-%m-%d %H:%M:%S')
        if pill_detail and pill_detail.crawled_at else None
    )
    updated_at = (
        pill_detail.updated_at.strftime('%Y-%m-%d %H:%M:%S')
        if pill_detail and pill_detail.updated_at else None
    )

    return {
        # ── 인식 메타 ──────────────────────────────────
        "status":           extra.get("status"),
        "method":           extra.get("method"),
        "confidence":       extra.get("confidence"),
        "tip":              extra.get("tip"),

        # ── 약품 기본 정보 ─────────────────────────────
        "dl_name":          m.get("dl_name"),
        "dl_name_en":       m.get("dl_name_en"),
        "dl_company":       m.get("dl_company"),
        "dl_company_en":    m.get("dl_company_en"),

        # ── 약품 외형 특성 ─────────────────────────────
        "drug_shape":       m.get("drug_shape"),
        "color_class1":     m.get("color_class1"),
        "color_class2":     m.get("color_class2"),
        "print_front":      m.get("print_front"),
        "print_back":       m.get("print_back"),
        "leng_long":        m.get("leng_long"),
        "leng_short":       m.get("leng_short"),
        "thick":            m.get("thick"),
        "form_code_name":   m.get("form_code_name"),
        "di_etc_otc_code":  m.get("di_etc_otc_code"),
        "chart":            m.get("chart"),

        # ── 약품 상세 정보 ─────────────────────────────
        "effect_text":      d.get("effect_text") or "효능 정보가 아직 없습니다. 복용 전 의사나 약사에게 꼭 확인하세요.",
        "usage_text":       d.get("usage_text")  or "복용 방법은 반드시 의사나 약사에게 확인하세요.",
        "warning_text":     d.get("warning_text"),
        "storage_text":     d.get("storage_text"),

        # ── 식별 정보 ──────────────────────────────────
        "item_seq":         m.get("item_seq"),
        "k_code":           m.get("k_code"),
        "di_class_no":      m.get("di_class_no"),
        "di_edi_code":      m.get("di_edi_code"),
        "dl_material":      m.get("dl_material"),

        # ── 시스템 정보 ────────────────────────────────
        "master_created_at": master_created_at,
        "crawled_at":        crawled_at,
        "updated_at":        updated_at,
    }


class PillMasterViewSet(ReadOnlyModelViewSet):
    """약품 마스터 정보 조회 전용 (생성·수정·삭제 비허용)"""
    queryset = PillMaster.objects.all()
    serializer_class = PillMasterSerializer


class PillAnalysisView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        image_obj = request.FILES.get('image')
        if not image_obj:
            return Response({"error": "이미지가 없습니다."}, status=400)

        try:
            # ── Step 1: 전처리 ───────────────────────────
            processed_image, ocr_img, color_pil = preprocess_for_inference(image_obj)

            if processed_image is None:
                return Response({
                    "status": "fail",
                    "error_type": "알약 미감지",
                    "message": "사진에서 알약을 찾지 못했습니다.",
                    "tip": "알약 하나를 밝은 배경 위에 놓고 정면으로 촬영해 주세요.",
                    "action": "다시 촬영"
                }, status=200)

            # ── Step 2: 듀얼 파이프라인 추론 ─────────────
            predicted_idx, confidence_score = predict_pill(
                processed_image,
                ocr_img=ocr_img,
                pill_data_map=id_mapping
            )
            confidence_percent = confidence_score * 100

            # ── Step 3: ResNet 성공 경로 ─────────────────
            if predicted_idx is not None:

                if confidence_percent < 10.0:
                    return Response({
                        "status": "fail",
                        "confidence": _confidence_label(confidence_percent),
                        "message": "알약이 아닌 것으로 판단됩니다. 사진을 다시 확인해 주세요.",
                        "error_type": "알약 아님",
                        "tip": "알약만 사진에 찍어 주세요. 손이나 다른 물건이 함께 나오지 않도록 해주세요.",
                        "action": "다시 촬영"
                    }, status=200)

                if confidence_percent < 30.0:
                    return Response({
                        "status": "fail",
                        "confidence": _confidence_label(confidence_percent),
                        "message": "알약은 맞지만 정확히 어떤 약인지 구별이 어렵습니다.",
                        "error_type": "인식 불명확",
                        "tip": "흰 종이 위에 알약을 올려놓고 밝은 곳에서 가까이 찍어 주세요.",
                        "action": "다시 촬영"
                    }, status=200)

                mapping_data = id_mapping.get(str(predicted_idx))
                if mapping_data:
                    target_item_seq = mapping_data.get('item_seq')
                    pill_master = PillMaster.objects.filter(item_seq=target_item_seq).first()
                    pill_detail = PillDetail.objects.filter(item_seq=target_item_seq).first()

                    response_data = _build_pill_response(pill_master, pill_detail, {
                        "status": "success",
                        "method": _method_label("ResNet"),
                        "ai_idx": str(predicted_idx),
                        "confidence": _confidence_label(confidence_percent),
                        "tip": None,
                    })
                    return Response(response_data, status=200)

                # ResNet은 맞췄으나 매핑 없음 → fallback으로 계속

            # ── Step 4: ResNet 실패 → DB 보완 검색 ───────
            # 각인(OCR) + 색상 + 모양을 조합해 DB에서 후보를 찾습니다.
            print(f"[FALLBACK] ResNet 실패 → 각인+색상+모양 보완 검색 시작")
            fallback = db_fallback_search(
                pil_image=color_pil,
                ocr_img=ocr_img,
                reader=reader,
                PillMaster=PillMaster,
            )

            if fallback:
                pill = fallback["pill"]
                method = fallback["method"]
                candidates = fallback["candidates"]
                pill_detail = PillDetail.objects.filter(item_seq=pill.item_seq).first()

                tip = (
                    f"비슷한 약이 {candidates}개 있으니 복용 전 반드시 약사나 의사에게 확인하세요."
                    if candidates > 1
                    else "복용 전 약 이름과 모양을 반드시 확인하세요."
                )

                response_data = _build_pill_response(pill, pill_detail, {
                    "status": "success",
                    "method": _method_label(method),
                    "ai_idx": None,
                    "confidence": _confidence_for_fallback(method),
                    "tip": tip,
                })
                return Response(response_data, status=200)

            # ── Step 5: 완전 실패 ─────────────────────────
            return Response({
                "status": "fail",
                "confidence": "알 수 없음",
                "message": "이 알약의 정보를 찾지 못했습니다.",
                "error_type": "정보 없음",
                "tip": "약에 새겨진 글자가 잘 보이도록 밝은 곳에서 다시 찍거나, 약 이름으로 직접 검색해 주세요.",
                "action": "직접 검색"
            }, status=200)

        except Exception as e:
            print(f"❌ 서버 내부 에러 발생: {str(e)}")
            return Response({"error": f"서비스 일시 오류: {str(e)}"}, status=500)

class PillSearchView(APIView):
    """
    약 이름 · 각인 텍스트로 검색합니다.
    GET /api/pills/search/?q=검색어
    """
    def get(self, request):
        q = request.query_params.get('q', '').strip()
        if not q:
            return Response({"error": "검색어(q)를 입력해 주세요."}, status=400)

        results = (
            PillMaster.objects.filter(dl_name__icontains=q)
            | PillMaster.objects.filter(dl_name_en__icontains=q)
            | PillMaster.objects.filter(print_front__icontains=q)
            | PillMaster.objects.filter(print_back__icontains=q)
        ).distinct()[:20]

        serializer = PillMasterSerializer(results, many=True)
        return Response({
            "query": q,
            "count": len(serializer.data),
            "results": serializer.data
        })