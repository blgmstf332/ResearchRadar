
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/processed"

FEATURES_FILE = DATA_DIR / "topic_year_features_v3_500.csv"
COUNTS_FILE = ROOT / "data/samples/topic_year_counts_500.csv"

TRAIN_FILE = DATA_DIR / "ml_training_v4_500.csv"
INFERENCE_FILE = DATA_DIR / "ml_inference_2025_v4_500.csv"

FEATURE_COLUMNS = [
    "growth_1y_pct",
    "growth_2y_pct",
    "growth_3y_pct",
    "growth_acceleration",
    "growth_stability",
]

EXPORT_COLUMNS = [
    "topic_id",
    "topic_name",
    "domain",
    "field",
    "subfield",
    "year",
    "publication_count",
    "prev_year_count",
    "count_2y_ago",
    "count_3y_ago",
] + FEATURE_COLUMNS

# 1. Verileri oku
features = pd.read_csv(
    FEATURES_FILE,
    dtype={"topic_id": str}
)

counts = pd.read_csv(
    COUNTS_FILE,
    dtype={"topic_id": str}
)

# 2. Veri dogrulama
required_features = set(EXPORT_COLUMNS)
required_counts = {
    "topic_id", "domain", "year", "publication_count"
}

if not required_features.issubset(features.columns):
    raise ValueError("Ozellik dosyasinda eksik sutun var.")

if not required_counts.issubset(counts.columns):
    raise ValueError("Yayin sayisi dosyasinda eksik sutun var.")

for name, data in [
    ("features", features),
    ("counts", counts)
]:
    if len(data) != 5000:
        raise ValueError(f"{name}: 5000 kayit bekleniyor.")

    if data["topic_id"].nunique() != 500:
        raise ValueError(f"{name}: 500 konu bekleniyor.")

    if data.duplicated(["topic_id", "year"]).any():
        raise ValueError(f"{name}: Tekrarlanan konu-yil.")

    if data[["topic_id", "year"]].isna().any().any():
        raise ValueError(f"{name}: Eksik konu veya yil.")

if counts["publication_count"].isna().any():
    raise ValueError("Eksik yayin sayisi bulundu.")

if (counts["publication_count"] < 0).any():
    raise ValueError("Negatif yayin sayisi bulundu.")

counts = counts.sort_values(["topic_id", "year"]).copy()

# 3. Sonraki yilin yayin sayisini hesapla
grouped = counts.groupby("topic_id")

counts["next_year"] = grouped["year"].shift(-1)

counts["next_count"] = (
    grouped["publication_count"].shift(-1)
)

# 4. Gelecek yilin buyume orani (%)
counts["future_growth_pct"] = (
    (
        counts["next_count"]
        - counts["publication_count"]
    )
    / counts["publication_count"].replace(0, np.nan)
    * 100
)

# Yillar ardisik degilse hedef olusturma
valid_next = counts["next_year"].eq(counts["year"] + 1)

counts.loc[
    ~valid_next, "future_growth_pct"
] = np.nan

# 5. Her domain ve hedef yil icin medyan buyume
# Ornek: 2023 ozellikleri -> 2024 buyumesi
counts["domain_median_future_growth"] = (
    counts.groupby(["domain", "next_year"])[
        "future_growth_pct"
    ].transform("median")
)

# 6. Ozellikler ve hedef hesaplama bilgilerini birlestir
label_columns = [
    "topic_id",
    "year",
    "domain",
    "publication_count",
    "next_year",
    "future_growth_pct",
    "domain_median_future_growth",
]

target_data = counts[label_columns].rename(
    columns={
        "domain": "actual_domain",
        "publication_count": "actual_count",
    }
)

df = features.merge(
    target_data,
    on=["topic_id", "year"],
    how="left",
    validate="one_to_one"
)

if len(df) != len(features):
    raise ValueError("Birlestirmede kayit sayisi degisti.")

if df["actual_count"].isna().any():
    raise ValueError("Bazi konu-yil sayilari eslesmedi.")

if not df["publication_count"].eq(
    df["actual_count"]
).all():
    raise ValueError("Yayin sayilari uyusmuyor.")

if not df["domain"].eq(df["actual_domain"]).all():
    raise ValueError("Domain bilgileri uyusmuyor.")

# 7. ML icin 2020-2024 yillarini sec
ml = df[df["year"].between(2020, 2024)].copy()

ml["split"] = ml["year"].map({
    2020: "train",
    2021: "train",
    2022: "train",
    2023: "validation",
    2024: "test",
})

# 8. Hedefi hesaplanabilen ve ozellikleri tam
# kayitlari tut
before = len(ml)

ml = ml.dropna(
    subset=FEATURE_COLUMNS + [
        "future_growth_pct",
        "domain_median_future_growth"
    ]
).copy()

if not ml["next_year"].eq(ml["year"] + 1).all():
    raise ValueError("Hedef yil uyusmazligi.")

print("Filtrelenen ML kaydi:", before - len(ml))

# 9. V3 hibrit hedef etiketi
relative_positive = (
    ml["future_growth_pct"]
    >= ml["domain_median_future_growth"] + 5
)

absolute_positive = (
    ml["future_growth_pct"] > 0
)

ml["target_hybrid_rise"] = (
    relative_positive & absolute_positive
).astype(int)

# 10. Gelecek yil bilgilerini disari aktarma!
# Yalnizca model girisleri, metadata ve hedef
# etiketini kaydet.
ml = ml[
    EXPORT_COLUMNS + ["split", "target_hybrid_rise"]
].copy()

ml = ml.sort_values(["year", "topic_id"])

# 11. 2025 verilerini 2026 tahmini icin ayir
inference = df[df["year"] == 2025].copy()

inference = inference.dropna(
    subset=FEATURE_COLUMNS
)

inference = inference[EXPORT_COLUMNS].copy()

inference = inference.sort_values("topic_id")

# 12. Kaydet
TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)

ml.to_csv(
    TRAIN_FILE,
    index=False,
    encoding="utf-8-sig"
)

inference.to_csv(
    INFERENCE_FILE,
    index=False,
    encoding="utf-8-sig"
)

# 13. Sonuc raporu
print("\n--- RESEARCHRADAR ML V4 / 500 TOPICS ---")

print("ML kaydi:", len(ml))
print("Benzersiz konu:", ml["topic_id"].nunique())

print("\nHedef dagilimi:")
print(ml["target_hybrid_rise"].value_counts())

print("\nTrain / Validation / Test:")
print(pd.crosstab(
    ml["split"],
    ml["target_hybrid_rise"]
))

print("\nPozitif oranlari:")

for split, group in ml.groupby("split"):
    rate = group["target_hybrid_rise"].mean() * 100
    print(f"{split}: %{rate:.2f}")

print("\n2025 inference kaydi:", len(inference))

print("\nEgitim dosyasi:", TRAIN_FILE)
print("Tahmin dosyasi:", INFERENCE_FILE)
