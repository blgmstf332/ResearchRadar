
import argparse
import os
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
    raise SystemExit("OpenAlex API anahtari bulunamadi.")

API_URL = "https://api.openalex.org/works"

START_YEAR = 2018
END_YEAR = 2025

INPUT_FILE = ROOT / "data/samples/topic_candidates_120.csv"

PROGRESS_FILE = (
    ROOT / "data/processed/topic_year_counts_progress.csv"
)

OUTPUT_FILE = (
    ROOT / "data/samples/topic_year_counts_120.csv"
)

PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# 2. Komut satiri ayarlari
parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=5)
args = parser.parse_args()

if not 1 <= args.limit <= 120:
    raise SystemExit("Limit 1 ile 120 arasinda olmali.")


# 3. API istek fonksiyonu
session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {API_KEY}"
})


def get_data(params):
    for attempt in range(5):
        try:
            response = session.get(
                API_URL,
                params=params,
                timeout=30
            )

            if response.status_code in (429, 500, 502, 503, 504):
                wait = 2 ** attempt
                print(f"API gecici hatasi: {response.status_code}")
                print(f"{wait} saniye sonra tekrar deneniyor.")
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError):
            wait = 2 ** attempt
            time.sleep(wait)

    raise RuntimeError("API istegi tekrar denemelere ragmen basarisiz.")


# 4. Konu listesini oku
topics_df = pd.read_csv(INPUT_FILE, dtype={"topic_id": str})

required_columns = {
    "topic_id", "topic_name", "domain", "field", "subfield"
}

if not required_columns.issubset(topics_df.columns):
    raise RuntimeError("Konu listesinde gerekli sutunlar eksik.")

if topics_df["topic_id"].duplicated().any():
    raise RuntimeError("Tekrarlanan topic_id bulundu.")

if len(topics_df) != 120:
    raise RuntimeError("Konu listesi 120 kayit icermiyor.")

selected_topics = topics_df.head(args.limit)


# 5. Daha once indirilen kayitlari oku
columns = [
    "topic_id", "topic_name", "domain", "field",
    "subfield", "year", "publication_count",
    "retrieved_at_utc"
]

if PROGRESS_FILE.exists():
    progress_df = pd.read_csv(
        PROGRESS_FILE,
        dtype={"topic_id": str}
    )

    if progress_df.duplicated(["topic_id", "year"]).any():
        raise RuntimeError("Checkpoint dosyasinda tekrar kayit var.")

else:
    progress_df = pd.DataFrame(columns=columns)


def topic_is_complete(topic_id):
    subset = progress_df[progress_df["topic_id"] == topic_id]

    expected = set(range(START_YEAR, END_YEAR + 1))
    actual = set(subset["year"].astype(int))

    return len(subset) == 8 and actual == expected


# 6. Tum konulari sirayla isle
for index, topic in selected_topics.iterrows():

    topic_id = topic["topic_id"]
    topic_name = topic["topic_name"]

    if topic_is_complete(topic_id):
        print(f"[SKIP] {topic_id} zaten indirildi.")
        continue

    # Yarim kalmis konu varsa yeniden isle
    progress_df = progress_df[
        progress_df["topic_id"] != topic_id
    ].copy()

    print(f"\n[{index + 1}/{args.limit}] {topic_name}")

    filters = (
        f"primary_topic.id:{topic_id},"
        f"publication_year:{START_YEAR}-{END_YEAR},"
        "type:article|preprint"
    )

    params = {
        "filter": filters,
        "group_by": "publication_year",
        "per_page": 100
    }

    data = get_data(params)

    counts = {
        int(item["key"]): int(item["count"])
        for item in data.get("group_by", [])
    }

    api_total = int(data["meta"]["count"])
    grouped_total = sum(counts.values())

    if grouped_total != api_total:
        raise RuntimeError(
            f"{topic_id}: API toplam sayisi ile "
            "yillik toplamlar uyusmuyor."
        )

    retrieved_at = datetime.now(timezone.utc).isoformat()

    rows = []

    for year in range(START_YEAR, END_YEAR + 1):
        rows.append({
            "topic_id": topic_id,
            "topic_name": topic_name,
            "domain": topic["domain"],
            "field": topic["field"],
            "subfield": topic["subfield"],
            "year": year,
            "publication_count": counts.get(year, 0),
            "retrieved_at_utc": retrieved_at
        })

    progress_df = pd.concat(
        [progress_df, pd.DataFrame(rows)],
        ignore_index=True
    )

    progress_df = progress_df.sort_values(
        ["topic_id", "year"]
    )

    # Her konu sonunda ilerlemeyi kaydet
    progress_df.to_csv(
        PROGRESS_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Yayin sayisi: {api_total}")
    print(f"Kaydedilen konu-yil satiri: {len(progress_df)}")

    time.sleep(0.4)


# 7. Sonuclari kontrol et
print("\n--- RESEARCHRADAR YILLIK VERI SONUCU ---")

completed_ids = [
    topic_id
    for topic_id in topics_df["topic_id"]
    if topic_is_complete(topic_id)
]

print("Tamamlanan konu:", len(completed_ids))
print("Kaydedilen satir:", len(progress_df))

if len(completed_ids) == 120:
    progress_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n120 konuluk yillik veri seti tamamlandi!")
    print("Dosya:", OUTPUT_FILE)
else:
    print("\nPilot veya kismi toplama tamamlandi.")
    print("Ilerleme dosyasi:", PROGRESS_FILE)
