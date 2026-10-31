
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/processed"
COUNTS_FILE = ROOT / "data/samples/topic_year_counts_120.csv"

V2_FILE = DATA_DIR / "ml_training_v2_relative.csv"
V3_FILE = DATA_DIR / "ml_training_v3_hybrid.csv"

# 1. V2 veri setini oku
ml = pd.read_csv(V2_FILE, dtype={"topic_id": str})

# Gercek yayin sayilarini oku
counts = pd.read_csv(
    COUNTS_FILE,
    dtype={"topic_id": str}
)

counts = counts.sort_values(["topic_id", "year"]).copy()

# 2. Sonraki yilin yayin sayisi
grouped = counts.groupby("topic_id")

counts["next_year"] = grouped["year"].shift(-1)
counts["next_count"] = grouped["publication_count"].shift(-1)

# 3. Gercek sonraki yil buyumesi
counts["future_growth_pct"] = (
    (counts["next_count"] - counts["publication_count"])
    / counts["publication_count"].replace(0, np.nan)
    * 100
)

# 4. V2 ile birlestir
df = ml.merge(
    counts[
        ["topic_id", "year", "next_year", "future_growth_pct"]
    ],
    on=["topic_id", "year"],
    how="left",
    validate="one_to_one"
)

if df["future_growth_pct"].isna().any():
    raise ValueError("Gelecek yil buyumesi eksik.")

if not (df["next_year"] == df["year"] + 1).all():
    raise ValueError("Ardisik yil kontrolu basarisiz.")

# 5. V2 pozitif olma kosulu zaten:
# Gelecek yil alan medyaninin 5 puan uzerinde buyume
relative_positive = df["target_relative_rise"] == 1

# V3: Hem gercek buyume pozitif
# hem de alan medyaninin 5 puan uzerinde
df["target_hybrid_rise"] = (
    relative_positive
    & (df["future_growth_pct"] > 0)
).astype(int)

# 6. Gelecek yil bilgilerini model girisinden cikar
# Bunlar yalnizca hedef olusturmak icindi.
df = df.drop(
    columns=[
        "target_relative_rise",
        "next_year",
        "future_growth_pct"
    ]
)

# 7. Veri dogrulama
if len(df) != len(ml):
    raise ValueError("Kayit sayisi degisti.")

if df.duplicated(["topic_id", "year"]).any():
    raise ValueError("Tekrarlanan konu-yil kaydi.")

# 8. Kaydet
df.to_csv(
    V3_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n--- RESEARCHRADAR HYBRID TARGET V3 ---")

print("Toplam ML kaydi:", len(df))
print("\nYeni hedef dagilimi:")
print(df["target_hybrid_rise"].value_counts())

print("\nTrain / Validation / Test:")
print(pd.crosstab(
    df["split"],
    df["target_hybrid_rise"]
))

print("\nPozitif oranlari:")

for split, group in df.groupby("split"):
    rate = group["target_hybrid_rise"].mean() * 100
    print(f"{split}: %{rate:.2f}")

print("\nKaydedildi:", V3_FILE)
