
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# 1. Proje ayarlari
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("OPENALEX_API_KEY")
if not API_KEY:
    raise SystemExit("API anahtari bulunamadi.")

INPUT_FILE = ROOT / "data/samples/topic_candidates_120.csv"
API_URL = "https://api.openalex.org/works"

# 2. Komut satiri ayarlari
parser = argparse.ArgumentParser()
parser.add_argument("--topics", type=int, default=2)
parser.add_argument("--per-year", type=int, default=5)
args = parser.parse_args()

if not 1 <= args.topics <= 120:
    raise ValueError("Konu sayisi 1-120 arasinda olmali.")

if not 1 <= args.per_year <= 100:
    raise ValueError("Yillik ornek 1-100 arasinda olmali.")

# 3. Kayit klasorleri
RAW_DIR = (
    ROOT / "data/raw/work_details"
    / f"sample_{args.per_year}"
)
PROCESSED_DIR = ROOT / "data/processed"
SAMPLE_DIR = ROOT / "data/samples"

for folder in (RAW_DIR, PROCESSED_DIR, SAMPLE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers["Authorization"] = f"Bearer {API_KEY}"

# 4. OpenAlex istek fonksiyonu
def api_get(params):
    for attempt in range(5):
        try:
            response = session.get(
                API_URL, params=params, timeout=60
            )

            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue

            response.raise_for_status()
            return response.json()

        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError):
            time.sleep(2 ** attempt)

    raise RuntimeError("OpenAlex istegi basarisiz.")


# 5. Abstract metnini yeniden olustur
def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return None

    words = []

    for word, positions in inverted_index.items():
        for position in positions:
            words.append((position, word))

    words.sort(key=lambda item: item[0])

    return " ".join(word for _, word in words)


# 6. Konulari oku
topics_df = pd.read_csv(
    INPUT_FILE,
    dtype={"topic_id": str}
)

if len(topics_df) != 120:
    raise RuntimeError("120 konuluk liste bulunamadi.")

selected_topics = topics_df.head(args.topics)

all_records = []

# 7. Konu ve yil bazinda makale topla
for topic in selected_topics.itertuples(index=False):
    topic_id = topic.topic_id

    print(f"\nKonu: {topic.topic_name}")

    for year in range(2018, 2026):
        output_path = RAW_DIR / f"{topic_id}_{year}.jsonl"

        if output_path.exists():
            print(f"  {year}: Kayit mevcut, atlaniyor.")

            with output_path.open(encoding="utf-8") as f:
                works = [json.loads(line) for line in f if line.strip()]

        else:
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
                raise RuntimeError("Beklenenden fazla makale geldi.")

            # Dosyayi once gecici adla yaz
            temp_path = output_path.with_suffix(".tmp")

            with temp_path.open("w", encoding="utf-8") as f:
                for work in works:
                    f.write(
                        json.dumps(work, ensure_ascii=False) + "\n"
                    )

            temp_path.replace(output_path)

            print(f"  {year}: {len(works)} makale alindi.")
            time.sleep(0.4)

        # 8. CSV icin duzlestir
        for work in works:
            abstract = reconstruct_abstract(
                work.get("abstract_inverted_index")
            )

            all_records.append({
                "work_id": work.get("id"),
                "topic_id": topic_id,
                "topic_name": topic.topic_name,
                "publication_year": work.get("publication_year"),
                "publication_date": work.get("publication_date"),
                "title": work.get("title"),
                "doi": work.get("doi"),
                "type": work.get("type"),
                "cited_by_count": work.get("cited_by_count"),
                "abstract": abstract,
                "has_abstract": bool(abstract),
                "counts_by_year_json": json.dumps(
                    work.get("counts_by_year") or []
                ),
                "is_retracted": work.get("is_retracted")
            })


# 9. Veri setini kaydet
df = pd.DataFrame(all_records)

if df["work_id"].duplicated().any():
    raise RuntimeError("Tekrarlanan work_id bulundu.")

processed_file = (
    PROCESSED_DIR
    / f"work_details_{args.topics}_topics_n{args.per_year}.csv"
)

df.to_csv(
    processed_file,
    index=False,
    encoding="utf-8-sig"
)

# GitHub icin sadece kucuk bir ornek
preview_file = SAMPLE_DIR / "work_details_preview.csv"

df.head(10).to_csv(
    preview_file,
    index=False,
    encoding="utf-8-sig"
)

# 10. Sonuc
print("\n--- RESEARCHRADAR WORK DETAILS ---")
print("Makale sayisi:", len(df))
print("Ozeti bulunan:", int(df["has_abstract"].sum()))
print("Ozeti bulunmayan:", int((~df["has_abstract"]).sum()))
print("Benzersiz makale:", df["work_id"].nunique())

print("\nYillik dagilim:")
print(df["publication_year"].value_counts().sort_index())

print("\nVeri dosyasi:", processed_file)
print("Ornek dosyasi:", preview_file)
print("Tamamlanma (UTC):", datetime.now(timezone.utc).isoformat())
