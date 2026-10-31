
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

# 1. Dosya yollari
ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = ROOT / "data/processed/topic_year_features_v1.csv"
WORKS_FILE = ROOT / "data/processed/work_details_120_topics_n30_cleaned.csv"
OUTPUT_FILE = ROOT / "data/processed/topic_year_features_v2.csv"

# 2. Verileri oku
features = pd.read_csv(
    FEATURE_FILE,
    dtype={"topic_id": str}
)

works = pd.read_csv(
    WORKS_FILE,
    dtype={"topic_id": str}
)

if features.duplicated(["topic_id", "year"]).any():
    raise ValueError("Tekrarlanan konu-yil ozellik kaydi.")

if works["work_id"].duplicated().any():
    raise ValueError("Tekrarlanan makale kimligi.")

# 3. Geri cekilmis yayinlari disla
retracted = (
    works["is_retracted"]
    .astype(str)
    .str.lower()
    .str.strip()
    .isin(["true", "1"])
)

print("Geri cekilmis makale:", int(retracted.sum()))

works = works.loc[~retracted].copy()

# 4. Her makalenin atif gecmisini incele
records = []
anomalous_events = 0
future_events = 0

for work in works.itertuples(index=False):

    pub_year = int(work.publication_year)
    raw = work.counts_by_year_json

    if pd.isna(raw) or str(raw).strip() == "":
        events = []
    else:
        events = json.loads(raw)

    if not isinstance(events, list):
        raise ValueError(
            f"Gecersiz atif listesi: {work.work_id}"
        )

    citations = defaultdict(int)

    for event in events:
        year = int(event["year"])
        count = int(event["cited_by_count"])

        if count < 0:
            raise ValueError("Negatif atif sayisi bulundu.")

        # Yayin oncesi atiflari kullanma
        if year < pub_year:
            anomalous_events += 1
            continue

        # 2026 veya sonraki atiflari dahil etme
        if year > 2025:
            future_events += 1
            continue

        citations[year] += count

    # Yayin yilini takip eden yildaki atif
    next_year_cites = citations.get(pub_year + 1, 0)

    # Yayin yili + takip eden yilin atiflari
    early_cites = (
        citations.get(pub_year, 0)
        + next_year_cites
    )

    records.append({
        "topic_id": work.topic_id,
        "publication_year": pub_year,
        "next_year_cites": next_year_cites,
        "early_cites": early_cites
    })

# 5. Konu ve yayin yilina gore toplulastir
citation_df = pd.DataFrame(records)

aggregated = (
    citation_df
    .groupby(["topic_id", "publication_year"], as_index=False)
    .agg(
        citation_sample_size=("next_year_cites", "size"),
        citation_velocity_1y=("next_year_cites", "mean"),
        citation_velocity_median=("next_year_cites", "median"),
        early_citation_mean=("early_cites", "mean")
    )
)

# 2022 yayinlarinin 2023 atiflari,
# 2023 yilinin ozelligidir.
aggregated["year"] = aggregated["publication_year"] + 1

aggregated = aggregated.drop(columns=["publication_year"])

# 6. Yayin buyumesi tablosuyla birlestir
result = features.merge(
    aggregated,
    on=["topic_id", "year"],
    how="left",
    validate="one_to_one"
)

if len(result) != len(features):
    raise RuntimeError("Konu-yil kayit sayisi degisti.")

# 7. Sonuclari kaydet
result.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n--- RESEARCHRADAR CITATION FEATURES ---")
print("Toplam konu-yil kaydi:", len(result))
print(
    "Atif ozelligi hesaplanan:",
    result["citation_velocity_1y"].notna().sum()
)
print("Yayin oncesi atif kayitlari:", anomalous_events)
print("Dislanan 2026+ atif kayitlari:", future_events)

print("\nIlk konunun sonuclari:")
first_topic = result["topic_id"].iloc[0]

print(
    result[result["topic_id"] == first_topic][
        [
            "year",
            "publication_count",
            "growth_1y_pct",
            "citation_sample_size",
            "citation_velocity_1y"
        ]
    ].to_string(index=False)
)

print("\nDosya:", OUTPUT_FILE)
