
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
)


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    print("Threshold:", round(threshold, 2))
    print("Macro-F1:", round(f1_score(
        y_true, predictions, average="macro"
    ), 4))
    print("Accuracy:", round(accuracy_score(
        y_true, predictions
    ), 4))
    print("Precision (sinif 1):", round(precision_score(
        y_true, predictions, zero_division=0
    ), 4))
    print("Recall (sinif 1):", round(recall_score(
        y_true, predictions, zero_division=0
    ), 4))
    print("Average Precision:", round(
        average_precision_score(y_true, probabilities), 4
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, predictions, labels=[0, 1]))


def main():
    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "processed" / "ml_training_v5_500_citations.csv"

    df = pd.read_csv(path)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()

    assert set(train_df["year"]) == {2020, 2021, 2022}
    assert set(val_df["year"]) == {2023}

    features = [
        "publication_count",
        "growth_1y_pct",
        "growth_2y_pct",
        "growth_3y_pct",
        "growth_acceleration",
        "growth_stability",
        "citation_sample_size",
        "citation_velocity_1y",
        "citation_velocity_median",
        "early_citation_mean",
    ]

    target = "target_hybrid_rise"

    # Gelecek yili tahmin etmek icin gecmis yillarla egitiyoruz.
    folds = [
        ([2020], 2021),
        ([2020, 2021], 2022),
    ]

    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.03, 0.07],
        "num_leaves": [7, 15, 31],
        "min_child_samples": [10, 30, 60],
    }

    thresholds = [step / 100 for step in range(10, 91, 5)]
    results = []

    combinations = list(ParameterGrid(param_grid))

    print("V5 LIGHTGBM HIPERPARAMETRE OPTIMIZASYONU")
    print("Ozellik sayisi:", len(features))
    print("Parametre kombinasyonu:", len(combinations))
    print("Zaman bazli fold sayisi:", len(folds))
    print("2024 test verisi KULLANILMIYOR.")
    print("=" * 65)

    for index, params in enumerate(combinations, start=1):
        fold_predictions = []

        for train_years, validation_year in folds:
            fold_train = train_df[
                train_df["year"].isin(train_years)
            ]
            fold_val = train_df[
                train_df["year"] == validation_year
            ]

            model = LGBMClassifier(
                **params,
                class_weight="balanced",
                random_state=42,
                verbosity=-1,
                n_jobs=-1,
            )

            model.fit(
                fold_train[features],
                fold_train[target],
            )

            probabilities = model.predict_proba(
                fold_val[features]
            )[:, 1]

            fold_predictions.append(
                (fold_val[target], probabilities)
            )

        # Her iki fold icin AYNI threshold kullaniliyor.
        for threshold in thresholds:
            scores = []

            for y_true, probabilities in fold_predictions:
                predictions = (
                    probabilities >= threshold
                ).astype(int)

                score = f1_score(
                    y_true,
                    predictions,
                    average="macro",
                )
                scores.append(score)

            mean_score = sum(scores) / len(scores)

            results.append({
                "params": params,
                "threshold": threshold,
                "cv_macro_f1": mean_score,
            })

        print(
            f"Tamamlandi: {index}/{len(combinations)}"
        )

    results.sort(
        key=lambda item: item["cv_macro_f1"],
        reverse=True,
    )

    print("\n" + "=" * 65)
    print("EN IYI 5 ZAMAN BAZLI CV SONUCU")
    print("=" * 65)

    for result in results[:5]:
        print(
            "CV Macro-F1:",
            round(result["cv_macro_f1"], 4),
            "| Threshold:",
            round(result["threshold"], 2),
            "| Parametreler:",
            result["params"],
        )

    best = results[0]
    best_params = best["params"]
    best_threshold = best["threshold"]

    print("\nSECILEN PARAMETRELER:")
    print(best_params)
    print("CV threshold:", round(best_threshold, 2))
    print("CV Macro-F1:", round(best["cv_macro_f1"], 4))

    # Tum 2020-2022 verisiyle yeniden egit.
    final_model = LGBMClassifier(
        **best_params,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )

    final_model.fit(
        train_df[features],
        train_df[target],
    )

    y_val = val_df[target]
    probabilities = final_model.predict_proba(
        val_df[features]
    )[:, 1]

    print("\n" + "=" * 65)
    print("2023 VALIDATION - THRESHOLD 0.50")
    print("=" * 65)
    evaluate(y_val, probabilities, 0.50)

    print("\n" + "=" * 65)
    print("2023 VALIDATION - CV ILE SECILEN THRESHOLD")
    print("=" * 65)
    evaluate(y_val, probabilities, best_threshold)

    # Onceki deneylerle karsilastirmak icin.
    # Bu sonuc validation verisinde optimize edildigi
    # icin iyimser olabilir.
    val_best_threshold = max(
        thresholds,
        key=lambda threshold: f1_score(
            y_val,
            (probabilities >= threshold).astype(int),
            average="macro",
        ),
    )

    print("\n" + "=" * 65)
    print("2023 VALIDATION - EN IYI THRESHOLD (KESIFSEL)")
    print("=" * 65)
    evaluate(y_val, probabilities, val_best_threshold)


if __name__ == "__main__":
    main()
