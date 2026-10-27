
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Proje ana klasörü
ROOT = Path(__file__).resolve().parents[2]

# API anahtarını yükle
load_dotenv(ROOT / ".env")
api_key = os.getenv("OPENALEX_API_KEY")

if not api_key:
    raise SystemExit("OpenAlex API anahtari bulunamadi.")

# Kullanıcıdan araştırma konusu al
query = input("Arastirmak istedigin konu: ").strip()

if not query:
    raise SystemExit("Konu bos olamaz.")

# OpenAlex Topics API
url = "https://api.openalex.org/topics"

params = {
    "search": query,
    "per_page": 10
}

headers = {
    "Authorization": f"Bearer {api_key}"
}

response = requests.get(
    url,
    params=params,
    headers=headers,
    timeout=30
)

response.raise_for_status()
data = response.json()

topics = data.get("results", [])

print("\n--- BULUNAN ARASTIRMA KONULARI ---")

if not topics:
    print("Eslesen konu bulunamadi.")

for i, topic in enumerate(topics, 1):
    print(f"\n{i}. {topic['display_name']}")
    print("Topic ID:", topic["id"])
    print("Toplam yayin:", topic["works_count"])
