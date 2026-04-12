import pandas as pd
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "data" / "interim" / "strategic_50_pills.csv"
OUTPUT_CSV = BASE_DIR / "data" / "interim" / "search_keywords.csv"


def extract_search_name(pill_name: str) -> str:
    if pd.isna(pill_name):
        return ""

    name = str(pill_name).strip()

    # 1. 쉼표 뒤 제거
    name = name.split(",")[0].strip()

    # 2. 포장정보 제거 (/PTP 등)
    name = name.split("/")[0].strip()

    # 3. 괄호 제거 (수출용, 성분명 등)
    name = re.sub(r"\([^)]*\)", "", name)

    # 4. 용량 제거 (200mg, 200 mg, 72.2mg, 5mg, 5 mg, 밀리그램 등)
    # 숫자 + 단위 (mg, g, mcg, 밀리그램)
    name = re.sub(r"\d+(?:\.\d+)?\s*(?:mg|g|mcg|밀리그램)", "", name, flags=re.IGNORECASE)

    # 5. 붙어있는 경우까지 제거 (예: 다오닐정5mg)
    name = re.sub(r"(?<=\D)\d+(?:\.\d+)?\s*(?:mg|g|mcg|밀리그램)", "", name, flags=re.IGNORECASE)

    # 6. 공백 정리
    name = re.sub(r"\s+", " ", name).strip()

    return name


def build_search_keywords(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv, encoding="utf-8-sig")

    if "pill_name" not in df.columns:
        raise ValueError("pill_name 컬럼이 없습니다.")

    keyword_df = df.copy()
    keyword_df["search_name"] = keyword_df["pill_name"].apply(extract_search_name)

    # 빈값 제거
    keyword_df = keyword_df[keyword_df["search_name"].astype(str).str.strip() != ""]

    # 중복 제거 (검색 키워드 기준)
    keyword_df = keyword_df.drop_duplicates(subset=["search_name"]).reset_index(drop=True)

    # 필요한 컬럼만 유지
    keyword_df = keyword_df[["pill_name", "search_name"]]

    # 저장
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    keyword_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return keyword_df


if __name__ == "__main__":
    result_df = build_search_keywords(INPUT_CSV, OUTPUT_CSV)

    print(f"검색 키워드 저장 완료: {OUTPUT_CSV}")
    print(result_df[["pill_name", "search_name"]].head(20).to_string(index=False))