
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Proje ana klasörünü bul
ROOT = Path(__file__).resolve().parents[2]

# .env dosyasından API anahtarını oku
load_dotenv(ROOT / ".env")
api_key = os.getenv("OPENALEX_API_KEY")

if not api_key:
    raise SystemExit("HATA: API anahtari bulunamadi!")

# OpenAlex API adresi
url = "https://api.openalex.org/works"

# 2024 yılından 3 makale getir
params = {
    "filter": "publication_year:2024",
    "per_page": 3,
    "select": "id,title,publication_year,cited_by_count"
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

print("OpenAlex API baglantisi basarili!")

for i, makale in enumerate(data["results"], start=1):
    print(f"\nMakale {i}")
    print("Baslik:", makale["title"])
    print("Yil:", makale["publication_year"])
    print("Atif:", makale["cited_by_count"])
