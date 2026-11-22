
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)

from final_v5_model import DATA_PATH, FEATURES, THRESHOLD


def evaluate(train, validation, features):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        max_features=0.7,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train[features],
        train["target_hybrid_rise"],
    )

    y_true = validation["target_hybrid_rise"]

    scores = model.predict_proba(
        validation[features]
    )[:, 1]

    predictions = (scores >= THRESHOLD).astype(int)

    return {
        "Macro-F1": f1_score(
            y_true, predictions, average="macro"
        ),
        "Accuracy": accuracy_score(
            y_true, predictions
        ),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "Average Precision": average_precision_score(
            y_true, scores
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    # Sadece 2020, 2021 ve 2022 verilerini kullan.
    df = df[
        (df["split"] == "train")
        & (df["year"].isin([2020, 2021, 2022]))
    ].copy()

    # Ayni yil ve alandaki buyume medyani.
    df["domain_median_growth"] = df.groupby(
        ["year", "domain"]
    )["growth_1y_pct"].transform("median")

    # Yeni ozellik.
    df["relative_growth_pp"] = (
        df["growth_1y_pct"]
        - df["domain_median_growth"]
    )

    old_features = FEATURES.copy()
    new_features = FEATURES + ["relative_growth_pp"]

    folds = [
        ([2020], 2021),
        ([2020, 2021], 2022),
    ]

    results = []

    print("=" * 65)
    print("V5 - ZAMAN BAZLI OZELLIK KARSILASTIRMASI")
    print("=" * 65)
    print("Threshold:", THRESHOLD)

    for train_years, validation_year in folds:
        train = df[
            df["year"].isin(train_years)
        ].copy()

        validation = df[
            df["year"] == validation_year
        ].copy()

        print("\n" + "=" * 65)
        print("Egitim yillari:", train_years)
        print("Validation yili:", validation_year)
        print("Egitim kaydi:", len(train))
        print("Validation kaydi:", len(validation))
        print("=" * 65)

        for model_name, features in [
            ("10 ozellik", old_features),
            ("11 ozellik", new_features),
        ]:
            metrics = evaluate(
                train, validation, features
            )

            print("\nModel:", model_name)

            for metric_name, value in metrics.items():
                print(
                    metric_name + ":",
                    round(value, 4)
                )

            results.append({
                "Validation_Yili": validation_year,
                "Model": model_name,
                **metrics,
            })

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("YILLARA GORE MACRO-F1 KARSILASTIRMASI")
    print("=" * 65)

    table = results_df.pivot(
        index="Validation_Yili",
        columns="Model",
        values="Macro-F1",
    )

    table["Fark"] = (
        table["11 ozellik"] - table["10 ozellik"]
    )

    print(table.round(4).to_string())

    print("\n" + "=" * 65)
    print("ORTALAMA SONUCLAR")
    print("=" * 65)

    averages = results_df.groupby("Model")[
        ["Macro-F1", "Accuracy", "Average Precision"]
    ].mean()

    print(averages.round(4).to_string())

    print("\n2023 ve 2024 verileri KULLANILMADI.")
    print("Kaydedilmis final model DEGISTIRILMEDI.")


if __name__ == "__main__":
    main()
