
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

training_file = ROOT / "data/processed/ml_training_v1.csv"
counts_file = ROOT / "data/samples/topic_year_counts_120.csv"

train = pd.read_csv(training_file, dtype={"topic_id": str})
counts = pd.read_csv(counts_file, dtype={"topic_id": str})

# Gercek yillik yayin sayilarini sirala
counts = counts.sort_values(["topic_id", "year"]).copy()

# Her konunun bir sonraki yildaki yayin sayisi
counts["next_year_count"] = (
    counts.groupby("topic_id")["publication_count"]
    .shift(-1)
)

# Sonraki yilin gercek buyume orani
counts["next_year_growth_pct"] = (
    (counts["next_year_count"] - counts["publication_count"])
    / counts["publication_count"].replace(0, np.nan)
    * 100
)

# ML tablosu ile birlestir
df = train.merge(
    counts[
        ["topic_id", "year", "next_year_growth_pct"]
    ],
    on=["topic_id", "year"],
    how="left",
    validate="one_to_one"
)

# Yillara gore hedef dagilimi
print("\n--- YILLARA GORE YUKSELIS ---")

for year, group in df.groupby("year"):
    n = len(group)
    positive = int(group["target_next_year_rise"].sum())

    print(
        f"{year} -> {year + 1} | "
        f"Toplam: {n} | "
        f"Yukselen: {positive} | "
        f"Oran: %{positive / n * 100:.1f} | "
        f"Medyan buyume: "
        f"%{group['next_year_growth_pct'].median():.2f}"
    )

# Farkli esiklerde yukselis oranini kontrol et
print("\n--- HEDEF ESIK KARSILASTIRMASI ---")

for threshold in [10, 15, 20, 25]:
    print(f"\nEsik: %{threshold}")

    for year, group in df.groupby("year"):
        rate = (
            group["next_year_growth_pct"] >= threshold
        ).mean() * 100

        print(f"  {year} -> {year + 1}: %{rate:.1f}")

# 2024 -> 2025 donemini ana alanlara ayir
print("\n--- 2024 -> 2025 DOMAIN ANALIZI ---")

last_year = df[df["year"] == 2024].copy()

summary = (
    last_year
    .groupby("domain")
    .agg(
        topic_count=("topic_id", "size"),
        rising_count=("target_next_year_rise", "sum"),
        median_growth=("next_year_growth_pct", "median")
    )
)

summary["rising_pct"] = (
    summary["rising_count"]
    / summary["topic_count"]
    * 100
)

print(summary.round(2).to_string())
