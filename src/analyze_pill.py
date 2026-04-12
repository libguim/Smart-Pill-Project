import pandas as pd
import os

def select_strategic_50(df):
    """
    기존의 전략적 구조를 유지하면서 50종을 확실히 추출하는 함수
    """
    final_list = []
    
    # [수정] 특정 색상/제형의 종수 확보를 위해 최소 이미지 제한을 100 -> 20으로 완화
    MIN_IMAGES = 20 

    # 전략 1: 색상 중심 (CNN 기본 분류용)
    colors = ['하양', '노랑', '주황', '분홍', '빨강', '갈색', '연두', '초록', '청록', '파랑', '보라', '자주']
    for color in colors:
        sample = df[df['color_class1'] == color].groupby('pill_name').filter(lambda x: len(x) >= MIN_IMAGES)
        if not sample.empty:
            top_pills = sample['pill_name'].value_counts().head(2).index
            final_list.append(df[df['pill_name'].isin(top_pills)].drop_duplicates('pill_name'))
    
    # 전략 2: 제형 및 모양 중심 (기하학적 특징용)
    shapes = ['장방형', '타원형', '삼각형', '사각형', '오각형', '육각형', '팔각형']
    for shape in shapes:
        sample = df[df['drug_shape'] == shape].groupby('pill_name').filter(lambda x: len(x) >= MIN_IMAGES)
        if not sample.empty:
            top_pills = sample['pill_name'].value_counts().head(2).index
            final_list.append(df[df['pill_name'].isin(top_pills)].drop_duplicates('pill_name'))

    # 전략 3: 각인이 뚜렷한 약 (OCR 성능 검증용)
    ocr_sample = df[df['print_front'].fillna('').str.len() >= 1].groupby('pill_name').filter(lambda x: len(x) >= MIN_IMAGES)
    if not ocr_sample.empty:
        top_ocr_pills = ocr_sample['pill_name'].value_counts().head(15).index
        final_list.append(df[df['pill_name'].isin(top_ocr_pills)].drop_duplicates('pill_name'))

    # 결과 통합 및 중복 제거
    result_df = pd.concat(final_list).drop_duplicates('pill_name')

    # [추가] 전략 기반 추출 후에도 50개가 부족할 경우 전체에서 무작위 보충
    if len(result_df) < 50:
        needed = 50 - len(result_df)
        remaining = df[~df['pill_name'].isin(result_df['pill_name'])].drop_duplicates('pill_name')
        if not remaining.empty:
            extra_pills = remaining.sample(n=min(len(remaining), needed), random_state=42)
            result_df = pd.concat([result_df, extra_pills])

    return result_df.head(50)

if __name__ == "__main__":
    # 1. 갱신된 상세 정보 CSV 로드
    CSV_PATH = 'data/interim/label_summary.csv'
    
    if not os.path.exists(CSV_PATH):
        print(f"❌ {CSV_PATH} 파일이 없습니다. preprocess.py를 먼저 실행하세요.")
    else:
        df = pd.read_csv(CSV_PATH)
        
        # 2. 전략적 추출 실행
        target_50 = select_strategic_50(df)
        
        if not target_50.empty:
            print(f"--- 🎯 전략 기반 타겟 {len(target_50)}종 선별 완료 ---")
            print(target_50[['pill_name', 'drug_shape', 'color_class1', 'print_front']].head(10))
            
            # [추가] 3. 최종 리스트 저장 (데이터 엔지니어링 파이프라인의 핵심)
            SAVE_DIR = 'data/interim'
            os.makedirs(SAVE_DIR, exist_ok=True)
            
            SAVE_PATH = os.path.join(SAVE_DIR, 'strategic_50_pills.csv')
            # 한글 깨짐 방지를 위해 utf-8-sig 사용
            target_50.to_csv(SAVE_PATH, index=False, encoding='utf-8-sig')
            
            print(f"\n✅ 저장 완료: {SAVE_PATH}")
        else:
            print("❌ 조건에 맞는 알약을 찾지 못했습니다.")