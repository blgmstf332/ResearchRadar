
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

file_path = ROOT / "data/samples/topic_year_counts_120.csv"

df = pd.read_csv(file_path, dtype={"topic_id": str})

required = {
    "topic_id", "domain", "year", "publication_count"
}

if not required.issubset(df.columns):
    raise ValueError("Gerekli sutunlar eksik.")

df = df.sort_values(["topic_id", "year"]).copy()

# Onceki yilin yayin sayisi
df["previous_count"] = (
    df.groupby("topic_id")["publication_count"].shift(1)
)

# Bir yillik buyume
df["growth_pct"] = (
    (df["publication_count"] - df["previous_count"])
    / df["previous_count"].replace(0, np.nan)
    * 100
)

# Her yil ve her ana bilim alani icin
# medyan buyumeyi hesapla
df["domain_median_growth"] = (
    df.groupby(["year", "domain"])["growth_pct"]
    .transform("median")
)

# Konunun kendi alanindan ne kadar
# daha hizli buyudugunu hesapla
df["relative_growth_pp"] = (
    df["growth_pct"] - df["domain_median_growth"]
)

# Sadece tahmin edilebilir hedef yillari
df = df[df["year"].between(2021, 2025)].copy()

print("\n--- BUYUME KARSILASTIRMASI ---")

for year, group in df.groupby("year"):
    valid = group.dropna(
        subset=["growth_pct", "relative_growth_pp"]
    )

    absolute_positive = (
        valid["growth_pct"] >= 15
    ).mean() * 100

    relative_positive = (
        valid["relative_growth_pp"] >= 5
    ).mean() * 100

    print(
        f"{year} | "
        f"Medyan: %{valid['growth_pct'].median():.2f} | "
        f"Mutlak >= %15: %{absolute_positive:.1f} | "
        f"Alan medyanindan +5 puan: %{relative_positive:.1f}"
    )

print("\n--- 2025 ALAN MEDYANLARI ---")

year_2025 = df[df["year"] == 2025]

print(
    year_2025.groupby("domain")["growth_pct"]
    .median()
    .round(2)
    .to_string()
)

print("\n--- 2025 EN YUKSEK GORELI BUYUME ---")

top = year_2025.sort_values(
    "relative_growth_pp", ascending=False
).head(10)

print(
    top[
        [
            "topic_name",
            "domain",
            "growth_pct",
            "domain_median_growth",
            "relative_growth_pp"
        ]
    ].round(2).to_string(index=False)
)
