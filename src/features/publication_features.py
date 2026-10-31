
from pathlib import Path

import pandas as pd
import numpy as np

# 1. Proje ana klasoru
ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/samples/topic_year_counts_120.csv"

OUTPUT_DIR = ROOT / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Veri setini oku
df = pd.read_csv(
    INPUT_FILE,
    dtype={"topic_id": str}
)

# 3. Veri dogrulama
required = {"topic_id", "year", "publication_count"}

if not required.issubset(df.columns):
    raise ValueError("Gerekli sutunlar eksik.")

if df.duplicated(["topic_id", "year"]).any():
    raise ValueError("Tekrarlanan konu-yil kaydi bulundu.")

if df["publication_count"].isna().any():
    raise ValueError("Eksik yayin sayisi bulundu.")

if (df["publication_count"] < 0).any():
    raise ValueError("Negatif yayin sayisi bulundu.")

df = df.sort_values(["topic_id", "year"]).copy()

if len(df) != 960 or df["topic_id"].nunique() != 120:
    raise ValueError("120 konu ve 960 satir bekleniyor.")

# 4. Onceki yillarin yayin sayilarini getir
grouped = df.groupby("topic_id")["publication_count"]

df["prev_year_count"] = grouped.shift(1)
df["count_2y_ago"] = grouped.shift(2)

# 5. Bir yillik buyume orani (%)
df["growth_1y_pct"] = (
    (
        df["publication_count"] - df["prev_year_count"]
    )
    / df["prev_year_count"].replace(0, np.nan)
    * 100
).round(2)

# 6. Iki yillik toplam buyume orani (%)
df["growth_2y_pct"] = (
    (
        df["publication_count"] - df["count_2y_ago"]
    )
    / df["count_2y_ago"].replace(0, np.nan)
    * 100
).round(2)

# 7. Sonuclari kaydet
output_file = OUTPUT_DIR / "topic_year_features_v1.csv"

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("\n--- RESEARCHRADAR FEATURE ENGINEERING ---")

print("Toplam kayit:", len(df))
print("Konu sayisi:", df["topic_id"].nunique())

print("\nIlk konunun yillik buyumesi:")
first_topic = df["topic_id"].iloc[0]

print(
    df[df["topic_id"] == first_topic][
        ["year", "publication_count",
         "growth_1y_pct", "growth_2y_pct"]
    ].to_string(index=False)
)

print("\nDosya kaydedildi:", output_file)
