
import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# 1. Proje ayarları
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("OPENALEX_API_KEY")

if not API_KEY:
    raise SystemExit("HATA: API anahtari bulunamadi.")

URL = "https://api.openalex.org/works"

TOPIC_ID = "T12072"
TOPIC_NAME = "Machine Learning and Algorithms"

START_YEAR = 2018
END_YEAR = 2025
TARGET_COUNT = 300

years = list(range(START_YEAR, END_YEAR + 1))

# 2. Kaydedilecek klasörler
raw_dir = ROOT / "data" / "raw"
sample_dir = ROOT / "data" / "samples"

raw_dir.mkdir(parents=True, exist_ok=True)
sample_dir.mkdir(parents=True, exist_ok=True)

# 3. Her yıl için örnek sayısını hesapla
base_count = TARGET_COUNT // len(years)
remainder = TARGET_COUNT % len(years)

all_works = []

for index, year in enumerate(years):

    # Toplam 300 kayıt yıl bazında paylaştırılıyor
    count = base_count + (1 if index < remainder else 0)

    print(f"\n{year} yilindan {count} makale isteniyor...")

    params = {
        "filter": (
            f"primary_topic.id:{TOPIC_ID},"
            f"publication_year:{year},"
            "type:article|preprint"
        ),
        "sample": count,
        "seed": 42 + year,
        "per_page": count,
        "select": (
            "id,title,doi,type,publication_year,"
            "publication_date,cited_by_count,"
            "counts_by_year,primary_topic,"
            "authorships,is_retracted"
        )
    }

    response = requests.get(
        URL,
        params=params,
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=60
    )

    response.raise_for_status()

    data = response.json()
    works = data.get("results", [])

    if len(works) != count:
        raise RuntimeError(
            f"{year} icin {count} kayit bekleniyordu, "
            f"{len(works)} geldi. Veri eksik kaydedilmeyecek."
        )

    all_works.extend(works)

    print(f"Alinan: {len(works)}")
    print(f"Toplam: {len(all_works)}")

    time.sleep(0.4)

# 4. OpenAlex kimliklerinin tekrarını kontrol et
ids = [work["id"] for work in all_works]

if len(ids) != len(set(ids)):
    raise RuntimeError("Tekrarlanan OpenAlex work_id bulundu.")

# 5. Ham veriyi JSONL olarak kaydet
raw_path = raw_dir / f"{TOPIC_ID}_pilot_300.jsonl"

with raw_path.open("w", encoding="utf-8") as file:
    for work in all_works:
        file.write(
            json.dumps(work, ensure_ascii=False) + "\n"
        )

# 6. Makaleleri tablo formatına dönüştür
rows = []

for work in all_works:
    authorships = work.get("authorships") or []

    institution_ids = set()

    for authorship in authorships:
        for institution in authorship.get("institutions", []):
            inst_id = institution.get("id")
            if inst_id:
                institution_ids.add(inst_id)

    rows.append({
        "work_id": work.get("id"),
        "topic_id": TOPIC_ID,
        "topic_name": TOPIC_NAME,
        "title": work.get("title"),
        "doi": work.get("doi"),
        "type": work.get("type"),
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "cited_by_count": work.get("cited_by_count"),
        "author_count_sampled": len(authorships),
        "institution_count_sampled": len(institution_ids),
        "is_retracted": work.get("is_retracted")
    })

df = pd.DataFrame(rows)

# 7. CSV olarak kaydet
csv_path = sample_dir / f"{TOPIC_ID}_pilot_300.csv"

df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)

# 8. Veri toplama metadata bilgisini kaydet
metadata = {
    "topic_id": TOPIC_ID,
    "topic_name": TOPIC_NAME,
    "start_year": START_YEAR,
    "end_year": END_YEAR,
    "sample_size": len(df),
    "sampling_method": "stratified_random_by_publication_year",
    "seed_rule": "42 + year",
    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    "filters": "primary_topic, publication_year, article|preprint"
}

metadata_path = sample_dir / f"{TOPIC_ID}_pilot_metadata.json"

with metadata_path.open("w", encoding="utf-8") as file:
    json.dump(metadata, file, ensure_ascii=False, indent=2)

# 9. Sonuçları göster
print("\n--- RESEARCHRADAR PILOT DATASET ---")
print("Konu:", TOPIC_NAME)
print("Toplam makale:", len(df))

print("\nYillara gore dagilim:")
print(df["publication_year"].value_counts().sort_index())

print("\nIlk 5 makale:")
print(df[["title", "publication_year", "cited_by_count"]].head())

print("\nCSV:", csv_path)
print("JSONL:", raw_path)
print("\nVeri toplama tamamlandi!")
