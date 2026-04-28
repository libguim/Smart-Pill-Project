import json
import os
import re

# 경로 설정
INPUT_JSON_PATH = os.path.join('pills', 'ai_models', 'pill_label_path_sharp_score.json')
SQL_FILE_PATH = 'pill.sql'
OUTPUT_MAP_PATH = os.path.join('pills', 'ai_models', 'pill_index_map.json')

def generate_pill_mapping():
    try:
        k_master_db = {}

        if os.path.exists(SQL_FILE_PATH):
            print(f"📖 {SQL_FILE_PATH} 분석 중...")
            with open(SQL_FILE_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # (1, 'K-037589', ...) 형태의 데이터 라인만 처리
                    if not (line.startswith('(') and 'K-' in line): continue
                    
                    # 쉼표로 분리하되 따옴표 내부 값 보존
                    # SQL 컬럼 순서: [1]k_code, [2]item_seq, [12]print_front, [13]print_back
                    parts = [p.strip().strip("'") for p in re.split(r",(?=(?:[^']*'[^']*')*[^']*$)", line.strip('(),;'))]
                    
                    if len(parts) > 1 and parts[1].startswith('K-'):
                        k_code = parts[1]
                        k_master_db[k_code] = {
                            "item_seq": parts[2] if len(parts) > 2 and parts[2] != 'NULL' else "",
                            "print_front": parts[10] if len(parts) > 10 and parts[10] != 'NULL' else "",
                            "print_back": parts[11] if len(parts) > 11 and parts[11] != 'NULL' else ""
                        }
            print(f"✅ SQL 딕셔너리 구축 완료: {len(k_master_db)}개 항목 확보")

        # 2. AI 모델 인덱스 데이터 로드
        if not os.path.exists(INPUT_JSON_PATH):
            print(f"🚨 {INPUT_JSON_PATH} 파일을 찾을 수 없습니다.")
            return

        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            data_list = raw_data.get("pill_label_path_sharp_score", [])

        # 3. 데이터 매핑 로직 (딕셔너리 기반 핀셋 수정)
        pill_index_map = {}
        success_count = 0

        for item in data_list:
            idx = str(item[0])
            k_code = str(item[1])
            
            # 딕셔너리에서 k_code로 즉시 조회 (Key lookup)
            info = k_master_db.get(k_code)
            
            if info:
                pill_index_map[idx] = {
                    "k_code": k_code,
                    "item_seq": info["item_seq"],
                    "print_front": info["print_front"],
                    "print_back": info["print_back"]
                }
                success_count += 1
            else:
                # SQL에 매칭 정보가 없을 경우 대비
                pill_index_map[idx] = {
                    "k_code": k_code,
                    "item_seq": re.sub(r'[^0-9]', '', k_code).zfill(9),
                    "print_front": "",
                    "print_back": ""
                }

        # 4. 결과 저장
        os.makedirs(os.path.dirname(OUTPUT_MAP_PATH), exist_ok=True)
        with open(OUTPUT_MAP_PATH, 'w', encoding='utf-8') as f:
            json.dump(pill_index_map, f, ensure_ascii=False, indent=4)

        print("-" * 50)
        print(f"🎉 생성 완료: {OUTPUT_MAP_PATH}")
        print(f"📊 총 인덱스: {len(pill_index_map)}개")
        print(f"📊 SQL 매칭 성공: {success_count}개")
        
        # 게루삼정(744번) 등 샘플 검증
        if "744" in pill_index_map:
            print(f"🔍 [검증] 744번 데이터: {pill_index_map['744']}")
        print("-" * 50)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    generate_pill_mapping()