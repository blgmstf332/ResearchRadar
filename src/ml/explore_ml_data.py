import pandas as pd

df = pd.read_csv("data/samples/T12072_yearly_counts.csv")

print("ResearchRadar - ML Veri İncelemesi")
print("-------------------------------")
print("Toplam satır:", len(df))
print("Toplam konu sayısı:", df["topic_id"].nunique())
print("İlk yıl:", df["year"].min())
print("Son yıl:", df["year"].max())


print("\nYıllık Yayın Sayısı Değişimleri")
print("-------------------------------")

df["yearly_change"] = df["publication_count"].diff()
df["growth_rate"] = df["publication_count"].pct_change() * 100

print(
    df[["year", "publication_count", "yearly_change", "growth_rate"]]
    .round(2)
    .to_string(index=False)
)

print("\nBir Sonraki Yılın Yükseliş Etiketi")
print("--------------------------------")

df = df.sort_values("year").copy()

df["next_year_count"] = df["publication_count"].shift(-1)

df["target_rising"] = (
    df["next_year_count"] > df["publication_count"]
).astype("Int64")

df.loc[df["next_year_count"].isna(), "target_rising"] = pd.NA

print(
    df[["year", "publication_count", "next_year_count", "target_rising"]]
    .to_string(index=False)
)

df.to_csv(
    "data/samples/T12072_ml_features.csv",
    index=False
)

print("\nML özellikleri CSV dosyasına kaydedildi.")