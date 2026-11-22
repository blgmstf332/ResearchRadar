
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def main():
    root = Path(__file__).resolve().parents[2]

    model_path = (
        root / "models" / "v5_random_forest.joblib"
    )

    data_path = (
        root / "data" / "processed"
        / "ml_training_v5_500_citations.csv"
    )

    # Kaydedilmis modeli yukle.
    # Model yeniden EGITILMEZ.
    bundle = joblib.load(model_path)

    model = bundle["model"]
    features = bundle["features"]
    threshold = bundle["threshold"]

    df = pd.read_csv(data_path)

    # Yalnizca 2024 test verisini sec.
    test_df = df[df["split"] == "test"].copy()

    assert len(test_df) > 0
    assert set(test_df["year"]) == {2024}

    missing = [
        feature for feature in features
        if feature not in test_df.columns
    ]

    if missing:
        raise ValueError(
            f"Eksik ozellikler: {missing}"
        )

    if test_df[features].isna().any().any():
        raise ValueError(
            "Test verisinde eksik sayisal deger var."
        )

    X_test = test_df[features]
    y_test = test_df["target_hybrid_rise"]

    # Modelin 1 sinifina verdigi skor.
    probabilities = model.predict_proba(X_test)[
        :, list(model.classes_).index(1)
    ]

    # Esik daha once 0.50 olarak sabitlendi.
    # Test sonucuna gore DEGISTIRILMEYECEK.
    predictions = (
        probabilities >= threshold
    ).astype(int)

    print("=" * 65)
    print("V5 RANDOM FOREST - 2024 FINAL TEST SONUCLARI")
    print("=" * 65)

    print("Model:", bundle["model_name"])
    print("Egitim yillari:", bundle["training_years"])
    print("Test yili: 2024")
    print("Test kaydi:", len(test_df))
    print("Ozellik sayisi:", len(features))
    print("Sabit threshold:", threshold)

    print("\nGERCEK SINIF DAGILIMI")
    print(y_test.value_counts().sort_index())

    print("\n" + "=" * 65)
    print("TEST METRIKLERI")
    print("=" * 65)

    print(
        "Accuracy:",
        round(accuracy_score(y_test, predictions), 4)
    )

    print(
        "Macro-F1:",
        round(
            f1_score(
                y_test, predictions, average="macro"
            ), 4
        )
    )

    print(
        "Precision (sinif 1):",
        round(
            precision_score(
                y_test, predictions, zero_division=0
            ), 4
        )
    )

    print(
        "Recall (sinif 1):",
        round(
            recall_score(
                y_test, predictions, zero_division=0
            ), 4
        )
    )

    print(
        "Average Precision:",
        round(
            average_precision_score(
                y_test, probabilities
            ), 4
        )
    )

    print("\nCONFUSION MATRIX")
    print("Satirlar: Gercek 0, Gercek 1")
    print("Sutunlar: Tahmin 0, Tahmin 1")
    print(
        confusion_matrix(
            y_test, predictions, labels=[0, 1]
        )
    )

    print("\nDETAYLI SINIFLANDIRMA RAPORU")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            zero_division=0,
        )
    )

    print("\nTest tamamlandi. Model degistirilmedi.")


if __name__ == "__main__":
    main()
