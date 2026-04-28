import torch
import json
import os
import django
from PIL import Image
from datetime import datetime
from django.forms.models import model_to_dict  # 객체를 딕셔너리로 변환하기 위해 추가

# 1. Django 환경 초기화
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from pills.models import PillMaster, PillDetail
from pills.ai_engine.loader import get_deployed_model

def run_unit_test(input_filename):
    # [경로 설정]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mapping_path = os.path.join(base_dir, 'pills', 'ai_models', 'pill_index_map.json')

    # [파일 탐색] 확장자 유연하게 처리
    image_source_dir = r'C:\backend\test_image'
    pure_name = os.path.splitext(input_filename)[0]
    valid_extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']
    image_path = next((os.path.join(image_source_dir, pure_name + ext) 
                       for ext in valid_extensions if os.path.exists(os.path.join(image_source_dir, pure_name + ext))), None)
    # image_path = next((os.path.join(base_dir, pure_name + ext) 
    #                    for ext in valid_extensions if os.path.exists(os.path.join(base_dir, pure_name + ext))), None)

    if not image_path:
        print(f"📍 파일을 찾을 수 없습니다: {os.path.join(image_source_dir, pure_name)} (확장자 포함)")
        return

    # 2. AI 추론 로직
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            id_mapping = json.load(f)
    except FileNotFoundError:
        print("❌ 매핑 파일을 찾을 수 없습니다.")
        return

    model, transform = get_deployed_model()
    model.eval()
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        conf, predicted_idx = torch.max(torch.nn.functional.softmax(outputs, dim=1), 1)
        idx = int(predicted_idx[0].item())
        confidence = float(conf[0].item())
        target_item_seq = id_mapping.get(str(idx), {}).get('item_seq', "unknown")

    # 3. DB 조회 (Master & Detail 전체 데이터)
    pill_master = PillMaster.objects.filter(item_seq=target_item_seq).first()
    pill_detail = PillDetail.objects.filter(item_seq=target_item_seq).first()

    # 날짜 포맷팅 함수
    def format_date(dt):
        return dt.strftime('%Y-%m-%d %H:%M:%S') if dt and isinstance(dt, datetime) else str(dt)

    # 4. 데이터 통합 (딕셔너리 병합 방식)
    if pill_master:
        # (1) 기본 정보 및 AI 분석 결과 초기화
        result = {
            "status": "success",
            "ai_id": str(idx),
            "confidence": f"{confidence * 100:.2f}%",
        }

        # (2) PillMaster 모든 필드 통합
        master_dict = model_to_dict(pill_master)
        if pill_master.created_at:
            master_dict['master_created_at'] = format_date(pill_master.created_at)
        result.update(master_dict)  # PillMaster 컬럼 전량 삽입

        # (3) PillDetail 모든 필드 통합 (중복 시 Detail 우선)
        if pill_detail:
            detail_dict = model_to_dict(pill_detail)
            if pill_detail.crawled_at:
                detail_dict['crawled_at'] = format_date(pill_detail.crawled_at)
            if pill_detail.updated_at:
                detail_dict['updated_at'] = format_date(pill_detail.updated_at)
            result.update(detail_dict)  # PillDetail 컬럼 전량 삽입 및 업데이트
        
        # 5. 결과 리포트 출력
        print(f"\n✅ 분석 이미지: {os.path.basename(image_path)}")
        print(f"--- AI 분석 리포트 ---")
        print(f"● 제품명: {result.get('dl_name', '정보 없음')}")
        print(f"● 모양: {result.get('drug_shape', 'N/A')} / 색상: {result.get('color_class1', 'N/A')}")
        print(f"● 각인: 앞({result.get('print_front', '없음')}) / 뒤({result.get('print_back', '없음')})")
        print(f"● 크기: {result.get('leng_long', '-')} x {result.get('thick', '-')} mm")
        print(f"● 신뢰도: {result['confidence']}")
        print(f"----------------------\n")
    else:
        result = {
            "status": "fail", 
            "message": "DB에서 정보를 찾을 수 없습니다.",
            "item_seq": target_item_seq
        }

    # 6. JSON 파일 저장
    save_path = os.path.join(base_dir, 'pills', 'ai_engine', 'response.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"📂 결과 저장 완료: {save_path}")

if __name__ == "__main__":
    # 테스트할 이미지 파일명 (확장자 제외 가능)
    run_unit_test("test_pill")