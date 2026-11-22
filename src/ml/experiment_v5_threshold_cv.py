
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    recall_score,
    confusion_matrix,
)

from final_v5_model import DATA_PATH, FEATURES

TARGET = "target_hybrid_rise"


def get_scores(df, train_years, validation_year):
    train = df[df["year"].isin(train_years)]
    validation = df[df["year"] == validation_year]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        max_features=0.7,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(train[FEATURES], train[TARGET])

    scores = model.predict_proba(
        validation[FEATURES]
    )[:, 1]

    return validation, scores


def metrics(validation, scores, threshold):
    actual = validation[TARGET].to_numpy()
    predictions = (scores >= threshold).astype(int)

    negative_growth = (
        (actual == 1)
        & (validation["growth_3y_pct"].to_numpy() < 0)
    )

    group_total = int(negative_growth.sum())
    group_correct = int(
        predictions[negative_growth].sum()
    )

    return {
        "Macro-F1": f1_score(
            actual, predictions, average="macro"
        ),
        "Recall": recall_score(
            actual, predictions, zero_division=0
        ),
        "Negatif_3y_Recall": (
            group_correct / group_total
            if group_total else 0
        ),
        "Yanlis_Alarm": int(
            ((actual == 0) & (predictions == 1)).sum()
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    # 2024 test verisi bu deneye dahil edilmiyor.
    df = df[df["year"].between(2020, 2023)].copy()

    assert set(df["year"]) == {
        2020, 2021, 2022, 2023
    }

    folds = [
        ([2020], 2021),
        ([2020, 2021], 2022),
    ]

    thresholds = np.round(
        np.arange(0.25, 0.651, 0.05), 2
    )

    results = []

    for train_years, val_year in folds:
        validation, scores = get_scores(
            df, train_years, val_year
        )

        for threshold in thresholds:
            result = metrics(
                validation, scores, threshold
            )

            results.append({
                "Yil": val_year,
                "Threshold": threshold,
                **result,
            })

    table = pd.DataFrame(results)

    summary = table.groupby(
        "Threshold"
    )[["Macro-F1", "Recall", "Negatif_3y_Recall"]].mean()

    print("=" * 65)
    print("2021 VE 2022 - KARAR ESIGI KARSILASTIRMASI")
    print("=" * 65)
    print(summary.round(4).to_string())

    best_threshold = float(
        summary["Macro-F1"].idxmax()
    )

    print("\nSecilen threshold:", best_threshold)
    print(
        "Ortalama CV Macro-F1:",
        round(
            summary.loc[best_threshold, "Macro-F1"], 4
        )
    )

    # Secilen esigi 2023 uzerinde kontrol et.
    # 2023 sonucu esik secmek icin kullanilmiyor.
    validation, scores = get_scores(
        df, [2020, 2021, 2022], 2023
    )

    print("\n" + "=" * 65)
    print("2023 VALIDATION KONTROLU")
    print("=" * 65)

    for threshold in sorted(set([
        0.50, best_threshold
    ])):
        result = metrics(
            validation, scores, threshold
        )

        print(f"\nThreshold: {threshold:.2f}")

        for key, value in result.items():
            print(f"{key}: {value:.4f}")

        actual = validation[TARGET]
        predictions = (
            scores >= threshold
        ).astype(int)

        print("Confusion Matrix:")
        print(
            confusion_matrix(
                actual, predictions, labels=[0, 1]
            )
        )

    print("\n2024 test verisi KULLANILMADI.")
    print("Kaydedilmis final model DEGISTIRILMEDI.")


if __name__ == "__main__":
    main()
