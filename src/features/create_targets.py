
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data/processed/topic_year_scores_v1.csv"
OUTPUT = ROOT / "data/processed"
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT, dtype={"topic_id": str})

df = df.sort_values(["topic_id", "year"]).copy()

if df.duplicated(["topic_id", "year"]).any():
    raise ValueError("Tekrarlanan konu-yil kaydi.")

# 1. Sonraki yilin yayin sayisini bul
grouped = df.groupby("topic_id")

df["next_year"] = grouped["year"].shift(-1)
df["next_year_count"] = grouped["publication_count"].shift(-1)

# 2. Sonraki yilin buyumesini hesapla
df["next_year_growth_pct"] = (
    (df["next_year_count"] - df["publication_count"])
    / df["publication_count"].replace(0, float("nan"))
    * 100
)

# 3. Gecerli hedef donemlerini belirle
valid_target = (
    df["year"].between(2020, 2024)
    & (df["next_year"] == df["year"] + 1)
    & (df["publication_count"] >= 50)
)

df["target_next_year_rise"] = pd.Series(
    pd.NA, index=df.index, dtype="Int64"
)

# %15 veya ustu buyume = 1
df.loc[valid_target, "target_next_year_rise"] = (
    df.loc[valid_target, "next_year_growth_pct"] >= 15
).astype(int)

# 4. Modele verilecek ozellikler
feature_cols = [
    "publication_count",
    "growth_1y_pct",
    "growth_2y_pct",
    "citation_velocity_1y",
    "citation_velocity_median",
    "early_citation_mean",
    "citation_sample_size"
]

metadata_cols = [
    "topic_id", "topic_name", "domain", "year"
]

# Eksik kritik ozellikleri bulunan kayitlari ayir
valid_features = (
    df[feature_cols].notna().all(axis=1)
    & (df["citation_sample_size"] >= 10)
)

# 5. Model egitim veri setini olustur
model_df = df.loc[
    valid_target & valid_features,
    metadata_cols + feature_cols + ["target_next_year_rise"]
].copy()

# Zaman bazli egitim/dogrulama/test ayrimi
model_df["split"] = model_df["year"].map({
    2020: "train",
    2021: "train",
    2022: "train",
    2023: "validation",
    2024: "test"
})

model_df.to_csv(
    OUTPUT / "ml_training_v1.csv",
    index=False,
    encoding="utf-8-sig"
)

# 6. 2025 icin henuz hedef etiketi yok
inference_df = df.loc[
    (df["year"] == 2025) & valid_features,
    metadata_cols + feature_cols
].copy()

inference_df.to_csv(
    OUTPUT / "ml_inference_2025_v1.csv",
    index=False,
    encoding="utf-8-sig"
)

# 7. Rapor
print("\n--- ML TARGET DATASET ---")
print("Egitim verisi satiri:", len(model_df))
print("2025 tahmin adayi:", len(inference_df))

print("\nHedef dagilimi:")
print(model_df["target_next_year_rise"].value_counts())

print("\nZaman bazli dagilim:")
print(pd.crosstab(
    model_df["split"],
    model_df["target_next_year_rise"]
))

print("\nKaydedildi: ml_training_v1.csv")
print("Kaydedildi: ml_inference_2025_v1.csv")
