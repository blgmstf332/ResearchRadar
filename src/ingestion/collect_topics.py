
import os
import time
from pathlib import Path
from datetime import datetime, timezone

import requests
import pandas as pd
from dotenv import load_dotenv

# 1. Proje ana klasörü
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("OPENALEX_API_KEY")

if not API_KEY:
    raise SystemExit("API anahtari bulunamadi!")

BASE_URL = "https://api.openalex.org"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

# 2. İlk deneme ayarları
TOPICS_PER_DOMAIN = 30
MIN_WORKS = 200
SEED = 42


# 3. OpenAlex API istek fonksiyonu
def get_data(endpoint, params=None):
    response = requests.get(
        BASE_URL + endpoint,
        params=params,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()
    return response.json()


# 4. OpenAlex ana alanlarını getir
domains_data = get_data(
    "/domains",
    {"per_page": 10}
)

domains = domains_data["results"]

all_topics = []

print("\n--- RESEARCHRADAR TOPIC COLLECTION ---")


# 5. Her ana alandan rastgele konu seç
for domain in domains:

    domain_id = domain["id"].split("/")[-1]
    domain_name = domain["display_name"]

    print(f"\nAna alan: {domain_name}")

    params = {
        "filter": (
            f"domain.id:{domain_id},"
            f"works_count:>{MIN_WORKS}"
        ),
        "sample": TOPICS_PER_DOMAIN,
        "seed": SEED + int(domain_id),
        "per_page": TOPICS_PER_DOMAIN,
        "select": (
            "id,display_name,works_count,"
            "domain,field,subfield"
        )
    }

    data = get_data("/topics", params)

    topics = data.get("results", [])

    for topic in topics:

        all_topics.append({
            "topic_id": topic["id"].split("/")[-1],
            "topic_name": topic["display_name"],
            "domain": topic["domain"]["display_name"],
            "field": topic["field"]["display_name"],
            "subfield": topic["subfield"]["display_name"],
            "works_count": topic["works_count"]
        })

    print(f"Alinan konu sayisi: {len(topics)}")

    time.sleep(0.3)


# 6. Verileri tabloya dönüştür
df = pd.DataFrame(all_topics)

# Tekrarlanan konuları kaldır
df = df.drop_duplicates(subset=["topic_id"])

df = df.sort_values(
    ["domain", "topic_name"]
).reset_index(drop=True)


# 7. CSV dosyasını kaydet
output_dir = ROOT / "data" / "samples"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "topic_candidates_120.csv"

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


# 8. Sonuçları göster
print("\n--- SECILEN ARASTIRMA KONULARI ---")

print(
    df[["topic_id", "topic_name", "domain"]].to_string(
        index=False
    )
)

print("\nToplam konu:", len(df))

print("\nAlanlara gore dagilim:")
print(df["domain"].value_counts())

print("\nDosya kaydedildi:", output_file)

print(
    "Indirme zamani (UTC):",
    datetime.now(timezone.utc).isoformat()
)

print("\n--- FIELD DAGILIMI ---")
print(df.groupby(["domain", "field"]).size().to_string())

if len(df) != 120:
    raise RuntimeError(
        f"120 konu bekleniyordu, {len(df)} bulundu."
    )

print("\n120 konuluk liste basariyla olusturuldu!")