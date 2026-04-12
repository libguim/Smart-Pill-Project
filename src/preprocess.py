import json
import os
import pandas as pd
from glob import glob

def summarize_labels(label_dir):
    summary_data = []
    # 모든 하위 폴더 내의 .json 파일을 찾습니다.
    json_files = glob(os.path.join(label_dir, "**", "*.json"), recursive=True)
    
    print(f"🔍 탐색 결과: 총 {len(json_files)}개의 JSON 파일을 발견했습니다.")
    print("⏳ 상세 정보(색상/모양/각인) 추출을 시작합니다...")

    for idx, file_path in enumerate(json_files):
        # 5,000개 단위로 진행 상황 표시
        if (idx + 1) % 5000 == 0:
            print(f"🔄 처리 중... ({idx + 1}/{len(json_files)})")
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if 'images' in data and 'annotations' in data:
                    img_info = data['images'][0]
                    # AI Hub 데이터 구조에 맞춘 상세 정보 추출
                    summary_data.append({
                        'file_name': img_info.get('file_name'),
                        'pill_name': img_info.get('dl_name'),
                        'drug_shape': img_info.get('drug_shape'),    # 모양 (원형, 장방형 등)
                        'color_class1': img_info.get('color_class1'), # 색상 (하양, 노랑 등)
                        'print_front': img_info.get('print_front'),   # 각인 (앞면)
                        'print_back': img_info.get('print_back'),     # 각인 (뒷면)
                        'width': img_info.get('width'),
                        'height': img_info.get('height'),
                        'bbox': data['annotations'][0].get('bbox')
                    })
        except Exception:
            continue
            
    return pd.DataFrame(summary_data)

if __name__ == "__main__":
    LABEL_PATH = 'data/raw/aihub_pill/labels/TL_1_단일'
    
    df = summarize_labels(LABEL_PATH)
    
    if not df.empty:
        os.makedirs('data/interim', exist_ok=True)
        # 덮어쓰기 저장
        df.to_csv('data/interim/label_summary.csv', index=False, encoding='utf-8-sig')
        print(f"✅ 완료! 상세 정보가 포함된 {len(df)}행의 CSV가 저장되었습니다.")
    else:
        print("❌ 데이터를 수집하지 못했습니다. 폴더 경로를 확인하세요.")