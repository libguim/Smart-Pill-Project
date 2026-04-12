import time
import io
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


BASE_DIR = Path(__file__).resolve().parent.parent
KEYWORD_CSV = BASE_DIR / "data" / "interim" / "search_keywords.csv"
SAVE_DIR = BASE_DIR / "data" / "raw" / "google_crawling" / "images"
META_CSV = BASE_DIR / "data" / "interim" / "google_image_metadata.csv"


def setup_driver(headless: bool = False) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")

    options.add_argument("--start-maximized")
    options.add_argument("--disable-gpu")
    options.add_argument("--lang=ko-KR")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/146.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(30)
    return driver


def sanitize_filename(name: str) -> str:
    name = name.replace("/", "_")
    invalid_chars = '\\:*?"<>|'
    for ch in invalid_chars:
        name = name.replace(ch, "_")
    return name.strip()


def load_queries(csv_path: Path, test_mode: bool = False) -> list[tuple[str, str, str]]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if "pill_name" not in df.columns or "search_name" not in df.columns:
        raise ValueError("pill_name 또는 search_name 컬럼이 없습니다.")

    rows = df[["pill_name", "search_name"]].dropna()

    if test_mode:
        rows = rows.head(3)

    return [
        (
            row["pill_name"].strip(),
            row["search_name"].strip(),
            f"{row['search_name'].strip()} 알약",
        )
        for _, row in rows.iterrows()
    ]


def open_google_images(driver: webdriver.Chrome, query: str):
    url = f"https://www.google.com/search?tbm=isch&q={quote(query)}"
    driver.get(url)
    time.sleep(3)


def captcha_or_block_detected(driver: webdriver.Chrome) -> bool:
    page_text = driver.page_source.lower()
    keywords = [
        "unusual traffic",
        "not a robot",
        "automated queries",
        "비정상적인 트래픽",
        "로봇이 아닙니다",
        "기계가 아닙니다",
    ]
    return any(k in page_text for k in keywords)


def wait_until_manual_check_finished(driver: webdriver.Chrome):
    while captcha_or_block_detected(driver):
        print("\n[주의] Google CAPTCHA 발생")
        input("브라우저에서 확인을 끝낸 뒤 Enter를 누르세요 → ")
        time.sleep(2)


def scroll_once(driver: webdriver.Chrome):
    body = driver.find_element(By.TAG_NAME, "body")
    body.send_keys(Keys.END)
    time.sleep(2)


def save_element_as_jpg(element, save_path: Path) -> bool:
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        png_bytes = element.screenshot_as_png
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        image.save(save_path, "JPEG", quality=90)
        return True
    except Exception as e:
        print(f"저장 실패: {save_path.name} / {e}")
        return False


def collect_visible_images(driver: webdriver.Chrome):
    """
    현재 화면에서 보이는 img 태그를 폭넓게 수집
    """
    results = []
    seen = set()

    imgs = driver.find_elements(By.TAG_NAME, "img")

    for img in imgs:
        try:
            if not img.is_displayed():
                continue

            size = img.size
            w = size.get("width", 0)
            h = size.get("height", 0)

            # 너무 작은 아이콘 제외
            if w < 60 or h < 60:
                continue

            src = img.get_attribute("src") or ""
            alt = img.get_attribute("alt") or ""

            key = (src, alt, w, h)
            if key in seen:
                continue
            seen.add(key)

            results.append((img, w, h, alt, src))

        except Exception:
            continue

    return results


def crawl_one_query(driver, pill_name, search_name, query, save_dir, max_images=5):
    folder_name = sanitize_filename(pill_name)
    pill_dir = save_dir / folder_name
    pill_dir.mkdir(parents=True, exist_ok=True)

    file_base = sanitize_filename(search_name)
    rows = []

    open_google_images(driver, query)
    wait_until_manual_check_finished(driver)

    # 결과가 덜 뜬 경우를 대비
    time.sleep(3)
    scroll_once(driver)
    wait_until_manual_check_finished(driver)
    time.sleep(2)

    candidates = collect_visible_images(driver)

    print(f"  - 수집된 이미지 후보 수: {len(candidates)}")

    if not candidates:
        return [{
            "pill_name": pill_name,
            "search_name": search_name,
            "query": query,
            "image_index": "",
            "saved_path": "",
            "status": "NO_VISIBLE_IMAGES",
        }]

    saved_count = 0

    for idx, (img, w, h, alt, src) in enumerate(candidates, start=1):
        if saved_count >= max_images:
            break

        try:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", img)
            time.sleep(0.3)

            file_path = pill_dir / f"{file_base}_{saved_count + 1:03d}.jpg"
            ok = save_element_as_jpg(img, file_path)

            if ok:
                saved_count += 1
                print(f"  - 저장 성공 {saved_count}: {file_path.name} ({w}x{h})")
                rows.append({
                    "pill_name": pill_name,
                    "search_name": search_name,
                    "query": query,
                    "image_index": saved_count,
                    "saved_path": str(file_path),
                    "status": "SAVED_VISIBLE_IMG",
                })

        except Exception as e:
            print(f"  - 후보 {idx} 실패: {e}")
            continue

    if not rows:
        rows.append({
            "pill_name": pill_name,
            "search_name": search_name,
            "query": query,
            "image_index": "",
            "saved_path": "",
            "status": "NO_IMAGE_SAVED",
        })

    return rows


def crawl_google_images(
    keyword_csv: Path,
    save_dir: Path,
    meta_csv: Path,
    headless: bool = False,
    test_mode: bool = False,
    max_images_per_keyword: int = 5
):
    queries = load_queries(keyword_csv, test_mode=test_mode)
    driver = setup_driver(headless=headless)
    all_meta = []

    try:
        for idx, (pill_name, search_name, query) in enumerate(queries, start=1):
            print(f"\n[{idx}/{len(queries)}] 검색 중: {query}")

            try:
                rows = crawl_one_query(
                    driver=driver,
                    pill_name=pill_name,
                    search_name=search_name,
                    query=query,
                    save_dir=save_dir,
                    max_images=max_images_per_keyword,
                )
                all_meta.extend(rows)

            except Exception as e:
                print(f"검색 실패: {query} / {e}")
                all_meta.append({
                    "pill_name": pill_name,
                    "search_name": search_name,
                    "query": query,
                    "image_index": "",
                    "saved_path": "",
                    "status": f"SEARCH_FAILED: {e}",
                })

            time.sleep(4)

    finally:
        driver.quit()

    meta_df = pd.DataFrame(all_meta)
    meta_csv.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(meta_csv, index=False, encoding="utf-8-sig")

    print(f"\n메타데이터 저장 완료: {meta_csv}")


if __name__ == "__main__":
    crawl_google_images(
        keyword_csv=KEYWORD_CSV,
        save_dir=SAVE_DIR,
        meta_csv=META_CSV,
        headless=False,
        test_mode=False,
        max_images_per_keyword=5,
    )