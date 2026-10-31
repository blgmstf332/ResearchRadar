
import pandas as pd

df = pd.read_csv(
    "data/processed/topic_year_scores_v1.csv"
)

df = df[df["year"].between(2020, 2024)].copy()

feature_cols = [
    "publication_count",
    "growth_1y_pct",
    "growth_2y_pct",
    "citation_velocity_1y",
    "citation_velocity_median",
    "early_citation_mean",
    "citation_sample_size"
]

low_publications = df["publication_count"] < 50
missing_features = df[feature_cols].isna().any(axis=1)
low_citation_sample = df["citation_sample_size"].fillna(0) < 10

excluded = (
    low_publications
    | missing_features
    | low_citation_sample
)

print("Toplam aday:", len(df))
print("50'den az yayini olan:", low_publications.sum())
print("Eksik ozelligi olan:", missing_features.sum())
print("Atif orneklemi 10'dan az:", low_citation_sample.sum())
print("Toplam dislanan (tekil):", excluded.sum())
print("Kalan kayit:", (~excluded).sum())

print("\nYillara gore dislanan:")
print(df.loc[excluded, "year"].value_counts().sort_index())
