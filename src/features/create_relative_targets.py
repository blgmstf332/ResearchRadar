
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/processed"

INPUT_FILE = DATA_DIR / "topic_year_features_v2.csv"
TRAIN_FILE = DATA_DIR / "ml_training_v2_relative.csv"
INFERENCE_FILE = DATA_DIR / "ml_inference_2025_v2.csv"

# 1. Mevcut ozellikleri oku
df = pd.read_csv(
    INPUT_FILE,
    dtype={"topic_id": str}
)

df = df.sort_values(["topic_id", "year"]).copy()

if len(df) != 960:
    raise ValueError("960 konu-yil kaydi bekleniyor.")

if df.duplicated(["topic_id", "year"]).any():
    raise ValueError("Tekrarlanan konu-yil kaydi bulundu.")

# 2. Her yilin alan medyan buyumesi
df["domain_median_growth"] = (
    df.groupby(["year", "domain"])["growth_1y_pct"]
    .transform("median")
)

# 3. Konunun alana gore buyume farki
df["relative_growth_pp"] = (
    df["growth_1y_pct"]
    - df["domain_median_growth"]
)

# 4. Sonraki yil bilgilerini olustur
grouped = df.groupby("topic_id")

df["next_year"] = grouped["year"].shift(-1)

df["next_relative_growth_pp"] = (
    grouped["relative_growth_pp"].shift(-1)
)

# 5. Sonraki yil alan medyanindan
# en az 5 yuzde puan hizli buyudu mu?
valid_target = (
    df["year"].between(2020, 2024)
    & (df["next_year"] == df["year"] + 1)
    & (df["publication_count"] >= 50)
    & df["next_relative_growth_pp"].notna()
)

df["target_relative_rise"] = pd.Series(
    pd.NA, index=df.index, dtype="Int64"
)

df.loc[valid_target, "target_relative_rise"] = (
    df.loc[valid_target, "next_relative_growth_pp"] >= 5
).astype(int)

# 6. Modelin gorebilecegi ozellikler
feature_cols = [
    "publication_count",
    "growth_1y_pct",
    "growth_2y_pct",
    "relative_growth_pp",
    "citation_velocity_1y",
    "citation_velocity_median",
    "early_citation_mean",
    "citation_sample_size"
]

metadata_cols = [
    "topic_id",
    "topic_name",
    "domain",
    "year"
]

valid_features = (
    df[feature_cols].notna().all(axis=1)
    & (df["citation_sample_size"] >= 10)
)

# 7. Etiketli ML veri seti
train_df = df.loc[
    valid_target & valid_features,
    metadata_cols
    + feature_cols
    + ["target_relative_rise"]
].copy()

train_df["split"] = train_df["year"].map({
    2020: "train",
    2021: "train",
    2022: "train",
    2023: "validation",
    2024: "test"
})

train_df.to_csv(
    TRAIN_FILE, index=False, encoding="utf-8-sig"
)

# 8. 2025 ozelliklerinden 2026 tahmin adayi
inference_df = df.loc[
    (df["year"] == 2025) & valid_features,
    metadata_cols + feature_cols
].copy()

inference_df.to_csv(
    INFERENCE_FILE, index=False, encoding="utf-8-sig"
)

# 9. Rapor
print("\n--- RELATIVE GROWTH ML DATASET ---")
print("Egitim veri seti:", len(train_df))
print("2025 tahmin adayi:", len(inference_df))

print("\nHedef dagilimi:")
print(train_df["target_relative_rise"].value_counts())

print("\nZaman bazli sinif dagilimi:")
print(pd.crosstab(
    train_df["split"],
    train_df["target_relative_rise"]
))

print("\nYillara gore pozitif oran:")
for year, group in train_df.groupby("year"):
    positive_rate = group["target_relative_rise"].mean() * 100
    print(f"{year} -> {year + 1}: %{positive_rate:.2f}")

print("\nDosyalar:")
print(TRAIN_FILE)
print(INFERENCE_FILE)
