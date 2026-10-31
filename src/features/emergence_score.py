
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data/processed/topic_year_features_v2.csv"
)

OUTPUT_FILE = (
    ROOT / "data/processed/topic_year_scores_v1.csv"
)

df = pd.read_csv(INPUT_FILE, dtype={"topic_id": str})

required = [
    "topic_id", "topic_name", "domain", "year",
    "growth_1y_pct", "citation_velocity_1y",
    "citation_sample_size"
]

if not set(required).issubset(df.columns):
    raise ValueError("Gerekli ozellik sutunlari eksik.")

if df.duplicated(["topic_id", "year"]).any():
    raise ValueError("Tekrarlanan konu-yil kaydi.")

# En az 10 makalelik atif orneklemi istiyoruz
eligible = (
    df["growth_1y_pct"].notna()
    & df["citation_velocity_1y"].notna()
    & (df["citation_sample_size"] >= 10)
)

df["growth_score"] = float("nan")
df["citation_score"] = float("nan")

# Her yil ve domain icinde karsilastirma
groups = ["year", "domain"]

df.loc[eligible, "growth_score"] = (
    df.loc[eligible]
    .groupby(groups)["growth_1y_pct"]
    .rank(pct=True)
    .mul(100)
)

df.loc[eligible, "citation_score"] = (
    df.loc[eligible]
    .groupby(groups)["citation_velocity_1y"]
    .rank(pct=True)
    .mul(100)
)

# %60 buyume + %40 atif
df["research_emergence_score"] = (
    0.60 * df["growth_score"]
    + 0.40 * df["citation_score"]
).round(2)

# Ilk deneysel yukselis adayi kurali
df["is_emerging_candidate"] = (
    (df["research_emergence_score"] >= 75)
    & (df["growth_1y_pct"] > 0)
    & (df["citation_velocity_1y"] > 0)
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n--- RESEARCHRADAR EMERGENCE SCORE ---")
print("Toplam konu-yil:", len(df))
print("Skor hesaplanan:", df["research_emergence_score"].notna().sum())

print("\n2025 YILININ EN YUKSEK 10 SKORU:")

top10 = (
    df[df["year"] == 2025]
    .sort_values("research_emergence_score", ascending=False)
    .head(10)
)

print(top10[
    [
        "topic_name",
        "domain",
        "growth_1y_pct",
        "citation_velocity_1y",
        "research_emergence_score"
    ]
].to_string(index=False))

print("\nDosya:", OUTPUT_FILE)
