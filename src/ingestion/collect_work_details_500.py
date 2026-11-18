
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# 1. Proje ayarlari
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("OPENALEX_API_KEY")

if not API_KEY:
    raise SystemExit("OPENALEX_API_KEY bulunamadi.")

OLD_TOPICS_FILE = (
    ROOT / "data/samples/topic_candidates_120.csv"
)

NEW_TOPICS_FILE = (
    ROOT / "data/samples/topic_candidates_500.csv"
)

API_URL = "https://api.openalex.org/works"

# 2. Komut satiri ayarlari
parser = argparse.ArgumentParser()

parser.add_argument("--topics", type=int, default=2)
parser.add_argument("--per-year", type=int, default=30)

args = parser.parse_args()

if not 1 <= args.topics <= 380:
    raise ValueError("Konu sayisi 1-380 arasinda olmali.")

if not 1 <= args.per_year <= 100:
    raise ValueError("Yillik ornek 1-100 arasinda olmali.")

# 3. Eski ve yeni konulari oku
old_topics = pd.read_csv(
    OLD_TOPICS_FILE,
    dtype={"topic_id": str}
)

all_topics = pd.read_csv(
    NEW_TOPICS_FILE,
    dtype={"topic_id": str}
)

old_ids = set(old_topics["topic_id"])
all_ids = set(all_topics["topic_id"])

if len(old_topics) != 120 or len(old_ids) != 120:
    raise ValueError("Eski 120 konu listesi gecersiz.")

if len(all_topics) != 500 or len(all_ids) != 500:
    raise ValueError("Yeni 500 konu listesi gecersiz.")

if not old_ids.issubset(all_ids):
    raise ValueError("Eski konulardan bazilari kaybolmus.")

# Yalnizca yeni 380 konu
new_topics = all_topics[
    ~all_topics["topic_id"].isin(old_ids)
].copy()

if len(new_topics) != 380:
    raise ValueError("380 yeni konu bekleniyor.")

selected_topics = new_topics.head(args.topics)

# 4. Ayri kayit klasoru
RAW_DIR = (
    ROOT / "data/raw/work_details_500_expansion"
    / f"sample_{args.per_year}"
)

RAW_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers["Authorization"] = f"Bearer {API_KEY}"

# 5. OpenAlex istek fonksiyonu
def api_get(params):
    for attempt in range(5):
        try:
            response = session.get(
                API_URL,
                params=params,
                timeout=60
            )

            if response.status_code == 429:
                message = response.text.lower()

                if "daily_budget" in message:
                    raise RuntimeError(
                        "Gunluk OpenAlex kotasi doldu. "
                        "Kayitlar korunuyor."
                    )

                wait = response.headers.get("Retry-After")

                try:
                    seconds = max(float(wait), 1)
                except (TypeError, ValueError):
                    seconds = 2 ** attempt

                time.sleep(seconds)
                continue

            if response.status_code in (500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue

            response.raise_for_status()
            return response.json()

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError
        ):
            if attempt == 4:
                raise

            time.sleep(2 ** attempt)

    raise RuntimeError(
        "OpenAlex istegi basarisiz. "
        "Kaydedilmis dosyalar korunuyor."
    )


# 6. Makale orneklerini al
def fetch_works(topic_id, year):
    params = {
        "filter": (
            f"primary_topic.id:{topic_id},"
            f"publication_year:{year},"
            "type:article|preprint"
        ),
        "sample": args.per_year,
        "seed": 42 + int(topic_id[1:]) + year,
        "per_page": args.per_year,
        "select": (
            "id,title,doi,type,publication_year,"
            "publication_date,cited_by_count,"
            "counts_by_year,abstract_inverted_index,"
            "authorships,primary_topic,is_retracted"
        )
    }

    data = api_get(params)
    works = data.get("results", [])

    if len(works) > args.per_year:
        raise ValueError("Beklenenden fazla makale geldi.")

    ids = [work.get("id") for work in works]

    if any(not work_id for work_id in ids):
        raise ValueError("Kimligi eksik makale bulundu.")

    if len(ids) != len(set(ids)):
        raise ValueError("Tekrarlanan makale kimligi.")

    for work in works:
        if work.get("publication_year") != year:
            raise ValueError("Makale yili uyusmuyor.")

        primary = work.get("primary_topic") or {}
        actual_topic = (
            str(primary.get("id") or "")
            .rstrip("/")
            .split("/")[-1]
        )

        if actual_topic != topic_id:
            raise ValueError("Makale konusu uyusmuyor.")

    return works


# 7. JSONL dosyasini guvenli yaz
def save_works(path, works):
    temporary = path.with_suffix(".tmp")

    with temporary.open("w", encoding="utf-8") as file:
        for work in works:
            file.write(
                json.dumps(work, ensure_ascii=False)
                + "\n"
            )

    temporary.replace(path)


# 8. Veri toplama
def main():
    print("\n--- RESEARCHRADAR 500 TOPIC EXPANSION ---")
    print("Eski konu:", len(old_ids))
    print("Yeni konu:", len(new_topics))
    print("Bu calismada secilen:", len(selected_topics))
    print("Yillik ornek:", args.per_year)
    print("Yillar: 2018-2025")

    new_requests = 0
    cached_files = 0
    total_works = 0

    for index, topic in enumerate(
        selected_topics.itertuples(index=False),
        start=1
    ):
        topic_id = topic.topic_id
        topic_total = 0

        for year in range(2018, 2026):
            output_path = (
                RAW_DIR / f"{topic_id}_{year}.jsonl"
            )

            if output_path.exists():
                # Mevcut dosyayi yeniden indirme.
                with output_path.open(
                    encoding="utf-8"
                ) as file:
                    works = [
                        json.loads(line)
                        for line in file
                        if line.strip()
                    ]

                if len(works) > args.per_year:
                    raise ValueError(
                        f"Gecersiz cache: {output_path}"
                    )

                cached_files += 1

            else:
                works = fetch_works(topic_id, year)
                save_works(output_path, works)

                new_requests += 1
                time.sleep(0.3)

            topic_total += len(works)
            total_works += len(works)

        print(
            f"[{index}/{len(selected_topics)}] "
            f"{topic_id}: {topic_total} makale"
        )

    print("\n--- SONUC ---")
    print("Islenen yeni konu:", len(selected_topics))
    print("Yeni API istegi:", new_requests)
    print("Onceden kayitli dosya:", cached_files)
    print("Toplam makale kaydi:", total_works)
    print("Ham veri klasoru:", RAW_DIR)


if __name__ == "__main__":
    try:
        main()
    finally:
        session.close()
