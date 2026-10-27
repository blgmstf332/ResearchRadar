
import os
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
    raise SystemExit("API anahtari bulunamadi!")

TOPIC_ID = "T12072"
TOPIC_NAME = "Machine Learning and Algorithms"

START_YEAR = 2018
END_YEAR = 2025

# 2. API isteği
url = "https://api.openalex.org/works"

params = {
    "filter": (
        f"primary_topic.id:{TOPIC_ID},"
        f"publication_year:{START_YEAR}-{END_YEAR},"
        "type:article|preprint"
    ),
    "group_by": "publication_year",
    "per_page": 100
}

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

response = requests.get(
    url,
    params=params,
    headers=headers,
    timeout=30
)

response.raise_for_status()
data = response.json()

# 3. Yıllık yayın sayılarını al
year_counts = {
    int(item["key"]): item["count"]
    for item in data.get("group_by", [])
}

# 4. Tablo oluştur
rows = []

for year in range(START_YEAR, END_YEAR + 1):
    rows.append({
        "topic_id": TOPIC_ID,
        "topic_name": TOPIC_NAME,
        "year": year,
        "publication_count": year_counts.get(year, 0),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat()
    })

df = pd.DataFrame(rows)

# 5. Veriyi CSV olarak kaydet
output_dir = ROOT / "data" / "samples"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "T12072_yearly_counts.csv"

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

# 6. Sonuçları göster
print("\n--- RESEARCHRADAR YILLIK YAYIN ANALIZI ---")
print("Konu:", TOPIC_NAME)

print("\n", df[["year", "publication_count"]].to_string(index=False))

print("\nToplam eslesen yayin:", data["meta"]["count"])
print("Yillik sayimlar toplami:", df["publication_count"].sum())

print("\nCSV kaydedildi:", output_file)
