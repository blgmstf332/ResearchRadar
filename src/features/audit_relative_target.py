
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

training = pd.read_csv(
    ROOT / "data/processed/ml_training_v2_relative.csv",
    dtype={"topic_id": str}
)

counts = pd.read_csv(
    ROOT / "data/samples/topic_year_counts_120.csv",
    dtype={"topic_id": str}
)

counts = counts.sort_values(["topic_id", "year"])

# Her konunun sonraki yil yayin sayisi
counts["next_count"] = (
    counts.groupby("topic_id")["publication_count"]
    .shift(-1)
)

# Gercek sonraki yil buyumesi
counts["future_growth_pct"] = (
    (counts["next_count"] - counts["publication_count"])
    / counts["publication_count"].replace(0, np.nan)
    * 100
)

# ML etiketleriyle birlestir
df = training.merge(
    counts[["topic_id", "year", "future_growth_pct"]],
    on=["topic_id", "year"],
    validate="one_to_one"
)

positive = df["target_relative_rise"] == 1

# Pozitif etiketli ama yayin sayisi artmayan konular
not_growing = (
    positive
    & (df["future_growth_pct"] <= 0)
)

print("\n--- RELATIVE TARGET AUDIT ---")

print("Toplam ML kaydi:", len(df))
print("Pozitif etiket:", int(positive.sum()))
print(
    "Pozitif ama gercekte buyumeyen:",
    int(not_growing.sum())
)

print("\nYillara gore durum:")

for year, group in df.groupby("year"):
    pos = group["target_relative_rise"] == 1

    not_growth = (
        pos & (group["future_growth_pct"] <= 0)
    )

    print(
        f"{year} -> {year + 1} | "
        f"Pozitif: {int(pos.sum())} | "
        f"Buyumeyen: {int(not_growth.sum())}"
    )

print("\nOrnek sorunlu konular:")

print(
    df.loc[not_growing, [
        "topic_name",
        "year",
        "future_growth_pct",
        "target_relative_rise"
    ]]
    .head(10)
    .round(2)
    .to_string(index=False)
)
