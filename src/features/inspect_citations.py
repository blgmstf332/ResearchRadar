
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

file_path = (
    ROOT / "data/processed/"
    "work_details_120_topics_n30_cleaned.csv"
)

df = pd.read_csv(
    file_path,
    usecols=[
        "work_id",
        "publication_year",
        "cited_by_count",
        "counts_by_year_json"
    ]
)

years = Counter()
invalid_json = 0
with_history = 0

for raw in df["counts_by_year_json"]:
    if pd.isna(raw):
        events = []
    else:
        try:
            events = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            invalid_json += 1
            continue

    if not isinstance(events, list):
        invalid_json += 1
        continue

    if events:
        with_history += 1

    for event in events:
        if not isinstance(event, dict):
            continue

        year = event.get("year")
        if isinstance(year, int):
            years[year] += 1

print("\n--- RESEARCHRADAR ATIF VERISI ---")
print("Toplam makale:", len(df))
print("Atif gecmisi bulunan:", with_history)
print("Atif gecmisi bos:", len(df) - with_history - invalid_json)
print("Gecersiz JSON:", invalid_json)

print("\nAtif yillarinin kapsami:")
for year, count in sorted(years.items()):
    print(f"{year}: {count} makalede atif kaydi")
