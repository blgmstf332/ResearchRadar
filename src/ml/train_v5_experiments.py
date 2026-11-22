
from pathlib import Path

import pandas as pd

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)


def main():
    root = Path(__file__).resolve().parents[2]

    data_path = (
        root / "data" / "processed"
        / "ml_training_v5_500_citations.csv"
    )

    df = pd.read_csv(data_path)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()

    assert set(train_df["year"]) == {2020, 2021, 2022}
    assert set(val_df["year"]) == {2023}

    target = "target_hybrid_rise"

    # Deney 1: Eski yayin ozellikleri
    old_publication_features = [
        "publication_count",
        "growth_1y_pct",
        "growth_2y_pct",
    ]

    # Deney 2: Tum yayin ozellikleri
    all_publication_features = old_publication_features + [
        "growth_3y_pct",
        "growth_acceleration",
        "growth_stability",
    ]

    # Deney 3: Yayin + atif ozellikleri
    all_features = all_publication_features + [
        "citation_sample_size",
        "citation_velocity_1y",
        "citation_velocity_median",
        "early_citation_mean",
    ]

    experiments = {
        "Deney 1 - Eski Yayin": old_publication_features,
        "Deney 2 - Tum Yayin": all_publication_features,
        "Deney 3 - Yayin ve Atif": all_features,
    }

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=10,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        ),
    }

    results = []

    print("V5 MODEL DENEYLERI BASLIYOR")
    print("Egitim kaydi:", len(train_df))
    print("Validation kaydi:", len(val_df))

    for experiment_name, features in experiments.items():

        print("\n" + "=" * 55)
        print(experiment_name)
        print("Ozellik sayisi:", len(features))
        print("Ozellikler:", features)
        print("=" * 55)

        X_train = train_df[features]
        y_train = train_df[target]

        X_val = val_df[features]
        y_val = val_df[target]

        for model_name, base_model in models.items():

            model = clone(base_model)
            model.fit(X_train, y_train)

            probabilities = model.predict_proba(X_val)[:, 1]
            predictions = (probabilities >= 0.50).astype(int)

            accuracy = accuracy_score(y_val, predictions)
            macro_f1 = f1_score(
                y_val, predictions, average="macro"
            )
            precision = precision_score(
                y_val, predictions, zero_division=0
            )
            recall = recall_score(
                y_val, predictions, zero_division=0
            )
            ap = average_precision_score(
                y_val, probabilities
            )
            cm = confusion_matrix(
                y_val, predictions, labels=[0, 1]
            )

            print("\n---", model_name, "---")
            print("Accuracy:", round(accuracy, 4))
            print("Macro-F1:", round(macro_f1, 4))
            print("Precision (sinif 1):", round(precision, 4))
            print("Recall (sinif 1):", round(recall, 4))
            print("Average Precision:", round(ap, 4))
            print("Confusion Matrix:")
            print(cm)


            # Validation uzerinde threshold optimizasyonu
            best_threshold = 0.50
            best_macro_f1 = macro_f1

            for step in range(10, 91, 5):
                threshold = step / 100
                candidate_predictions = (
                    probabilities >= threshold
                ).astype(int)

                score = f1_score(
                    y_val,
                    candidate_predictions,
                    average="macro",
                )

                if score > best_macro_f1:
                    best_macro_f1 = score
                    best_threshold = threshold

            best_predictions = (
                probabilities >= best_threshold
            ).astype(int)

            print("\n--- OPTIMIZE EDILMIS SONUC ---")
            print("En iyi threshold:", round(best_threshold, 2))
            print("En iyi Macro-F1:", round(best_macro_f1, 4))
            print(
                "Accuracy:",
                round(accuracy_score(y_val, best_predictions), 4)
            )
            print(
                "Precision (sinif 1):",
                round(
                    precision_score(
                        y_val, best_predictions, zero_division=0
                    ), 4
                )
            )
            print(
                "Recall (sinif 1):",
                round(
                    recall_score(
                        y_val, best_predictions, zero_division=0
                    ), 4
                )
            )
            print("Confusion Matrix:")
            print(confusion_matrix(y_val, best_predictions))


            results.append({
                "Deney": experiment_name,
                "Model": model_name,
                "Macro-F1": round(macro_f1, 4),
                "Best_Threshold": round(best_threshold, 2),
                "Optimized_Macro-F1": round(best_macro_f1, 4),
                "Accuracy": round(accuracy, 4),
                "Precision_1": round(precision, 4),
                "Recall_1": round(recall, 4),
                "AP": round(ap, 4),
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
    "Optimized_Macro-F1", ascending=False
    )

    print("\n" + "=" * 70)
    print("TUM MODELLERIN VALIDATION KARSILASTIRMASI")
    print("Threshold: 0.50")
    print("=" * 70)
    print(results_df.to_string(index=False))

    print("\nEn iyi model:")
    print(results_df.iloc[0].to_string())


if __name__ == "__main__":
    main()
