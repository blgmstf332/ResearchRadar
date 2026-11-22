import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.ml.final_v5_model import DATA_PATH, FEATURES

df = pd.read_csv(DATA_PATH)
df = df[df["year"].between(2020, 2023)].copy()

df["domain_growth_percentile"] = (
    df.groupby(["year", "domain"])["growth_1y_pct"]
    .rank(pct=True, method="average")
)

folds = [
    ([2020], 2021),
    ([2020, 2021], 2022),
    ([2020, 2021, 2022], 2023),
]

results = []

for years, val_year in folds:
    train = df[df["year"].isin(years)]
    val = df[df["year"] == val_year]

    for seed in [42, 123, 777]:
        for name, features in [
            ("10 ozellik", FEATURES),
            ("11 ozellik", FEATURES + ["domain_growth_percentile"]),
        ]:
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                max_features=0.7,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )

            model.fit(
                train[features],
                train["target_hybrid_rise"],
            )

            predictions = (
                model.predict_proba(val[features])[:, 1] >= 0.50
            ).astype(int)

            results.append({
                "Yil": val_year,
                "Seed": seed,
                "Model": name,
                "Macro_F1": f1_score(
                    val["target_hybrid_rise"],
                    predictions,
                    average="macro",
                ),
            })

result = pd.DataFrame(results)

table = result.pivot(
    index=["Yil", "Seed"],
    columns="Model",
    values="Macro_F1",
)

table["Fark"] = table["11 ozellik"] - table["10 ozellik"]

print("\nYILLARA VE SEED DEGERLERINE GORE SONUCLAR")
print(table.round(4).to_string())

print("\nYILLIK ORTALAMALAR")
print(table.groupby(level="Yil").mean().round(4).to_string())

print("\n11 OZELLIK KAC DENEYDE DAHA IYI?")
print(int((table["Fark"] > 0).sum()), "/ 9")

print("\n2024 test verisi kullanilmadi.")
