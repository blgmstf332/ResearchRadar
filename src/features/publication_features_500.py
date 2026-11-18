
from pathlib import Path

import numpy as np
import pandas as pd

# 1. Dosya yollari
ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/samples/topic_year_counts_500.csv"
OUTPUT_FILE = ROOT / "data/processed/topic_year_features_v3_500.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# 2. Veri setini oku
df = pd.read_csv(
    INPUT_FILE,
    dtype={"topic_id": str}
)

# 3. Veri dogrulama
required = {"topic_id", "year", "publication_count"}

if not required.issubset(df.columns):
    raise ValueError("Gerekli sutunlar eksik.")

if len(df) != 5000 or df["topic_id"].nunique() != 500:
    raise ValueError("500 konu ve 5000 kayit bekleniyor.")

if df.duplicated(["topic_id", "year"]).any():
    raise ValueError("Tekrarlanan konu-yil kaydi var.")

if df[list(required)].isna().any().any():
    raise ValueError("Gerekli alanlarda eksik veri var.")

if (df["publication_count"] < 0).any():
    raise ValueError("Negatif yayin sayisi bulundu.")

expected_years = set(range(2016, 2026))

for topic_id, group in df.groupby("topic_id"):
    if set(group["year"]) != expected_years:
        raise ValueError(
            f"{topic_id}: 2016-2025 yillari eksik."
        )

df = df.sort_values(["topic_id", "year"]).copy()

# 4. Gecmis yillarin yayin sayilari
grouped = df.groupby("topic_id")["publication_count"]

df["prev_year_count"] = grouped.shift(1)
df["count_2y_ago"] = grouped.shift(2)
df["count_3y_ago"] = grouped.shift(3)

# 5. Bir yillik buyume (%)
df["growth_1y_pct"] = (
    (df["publication_count"] - df["prev_year_count"])
    / df["prev_year_count"].replace(0, np.nan)
    * 100
)

# 6. Iki yillik toplam buyume (%)
df["growth_2y_pct"] = (
    (df["publication_count"] - df["count_2y_ago"])
    / df["count_2y_ago"].replace(0, np.nan)
    * 100
)

# 7. Uc yillik toplam buyume (%)
df["growth_3y_pct"] = (
    (df["publication_count"] - df["count_3y_ago"])
    / df["count_3y_ago"].replace(0, np.nan)
    * 100
)

# 8. Buyume ivmesi
previous_growth = (
    df.groupby("topic_id")["growth_1y_pct"].shift(1)
)

df["growth_acceleration"] = (
    df["growth_1y_pct"] - previous_growth
)

# 9. Buyume istikrari
# Son 3 yillik buyume oranlarinin standart sapmasi.
df["growth_stability"] = (
    df.groupby("topic_id")["growth_1y_pct"]
    .transform(
        lambda s: s.rolling(
            window=3,
            min_periods=3
        ).std(ddof=0)
    )
)

# 10. Sayilari yuvarla
feature_columns = [
    "growth_1y_pct",
    "growth_2y_pct",
    "growth_3y_pct",
    "growth_acceleration",
    "growth_stability",
]

df[feature_columns] = df[feature_columns].round(2)

# 11. Sonuclari kaydet
df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n--- RESEARCHRADAR 500 TOPIC FEATURES ---")

print("Toplam kayit:", len(df))
print("Benzersiz konu:", df["topic_id"].nunique())
print("Yil araligi:", df["year"].min(), "-", df["year"].max())

print("\nOzelliklerdeki eksik degerler:")
print(df[feature_columns].isna().sum().to_string())

first_topic = df["topic_id"].iloc[0]

print(f"\nOrnek konu: {first_topic}")

print(
    df[df["topic_id"] == first_topic][
        ["year", "publication_count"] + feature_columns
    ].to_string(index=False)
)

print("\nDosya kaydedildi:", OUTPUT_FILE)
