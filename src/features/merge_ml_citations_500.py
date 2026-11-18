
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/processed"

TRAIN_FILE = DATA_DIR / "ml_training_v4_500.csv"
INFERENCE_FILE = DATA_DIR / "ml_inference_2025_v4_500.csv"
CITATION_FILE = DATA_DIR / "topic_year_features_v4_500_citations.csv"

TRAIN_OUTPUT = DATA_DIR / "ml_training_v5_500_citations.csv"
INFERENCE_OUTPUT = DATA_DIR / "ml_inference_2025_v5_500_citations.csv"

CITATION_COLUMNS = [
    "citation_sample_size",
    "citation_velocity_1y",
    "citation_velocity_median",
    "early_citation_mean",
]

GROWTH_COLUMNS = [
    "growth_1y_pct",
    "growth_2y_pct",
    "growth_3y_pct",
    "growth_acceleration",
    "growth_stability",
]

# 1. Verileri oku
train = pd.read_csv(TRAIN_FILE, dtype={"topic_id": str})
inference = pd.read_csv(INFERENCE_FILE, dtype={"topic_id": str})
citations = pd.read_csv(CITATION_FILE, dtype={"topic_id": str})

# 2. Tekrar kontrolu
for name, df in [
    ("train", train),
    ("inference", inference),
    ("citations", citations),
]:
    if df.duplicated(["topic_id", "year"]).any():
        raise ValueError(f"{name}: Tekrar eden konu-yil.")

# 3. Atif sutunlarini al
citation_data = citations[
    ["topic_id", "year"] + CITATION_COLUMNS
].copy()

# 4. Birlestirme fonksiyonu
def merge_citations(df, name):
    result = df.merge(
        citation_data,
        on=["topic_id", "year"],
        how="left",
        validate="one_to_one"
    )

    if len(result) != len(df):
        raise ValueError(f"{name}: Kayit sayisi degisti.")

    missing = result[CITATION_COLUMNS].isna().sum()

    if missing.any():
        print(f"\n{name} eksik atif degerleri:")
        print(missing)
        raise ValueError("Eksik atif ozellikleri bulundu.")

    numeric_columns = GROWTH_COLUMNS + CITATION_COLUMNS

    if not np.isfinite(
        result[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(f"{name}: Gecersiz sayisal deger.")

    if not result["citation_sample_size"].between(1, 30).all():
        raise ValueError("Atif orneklem buyuklugu gecersiz.")

    # Gelecek yila ait bilgiler model girdisine giremez.
    forbidden = {
        "next_year",
        "next_count",
        "future_growth_pct",
        "domain_median_future_growth",
    }

    if forbidden.intersection(result.columns):
        raise ValueError("Gelecek yil bilgisi bulundu!")

    return result


# 5. Egitim ve inference verilerini birlestir
train_v5 = merge_citations(train, "training")
inference_v5 = merge_citations(inference, "inference")

# 6. Kritik kontroller
if len(train_v5) != 2495:
    raise ValueError("2495 ML kaydi bekleniyor.")

if len(inference_v5) != 499:
    raise ValueError("499 inference kaydi bekleniyor.")

if "target_hybrid_rise" not in train_v5.columns:
    raise ValueError("Hedef etiketi eksik.")

if "target_hybrid_rise" in inference_v5.columns:
    raise ValueError("Inference dosyasinda hedef olmamali.")

# 7. Dosyalari kaydet
train_v5.to_csv(
    TRAIN_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

inference_v5.to_csv(
    INFERENCE_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

# 8. Sonuc
print("\n--- RESEARCHRADAR ML V5 ---")

print("Egitim + validation + test:", len(train_v5))
print("2025 inference:", len(inference_v5))

print("\nVeri bolumleri:")
print(pd.crosstab(
    train_v5["split"],
    train_v5["target_hybrid_rise"]
))

print("\nAtif ozelliklerinde eksik deger:")
print(train_v5[CITATION_COLUMNS].isna().sum())

print("\nML ozellikleri:")
for column in GROWTH_COLUMNS + CITATION_COLUMNS:
    print("-", column)

print("\nEgitim dosyasi:", TRAIN_OUTPUT)
print("Inference dosyasi:", INFERENCE_OUTPUT)
