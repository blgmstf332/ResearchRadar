
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
)

from final_v5_model import DATA_PATH, FEATURES, THRESHOLD


def evaluate(train, validation, features, name):
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

    scores = model.predict_proba(
        validation[features]
    )[:, 1]

    predictions = (scores >= THRESHOLD).astype(int)
    actual = validation["target_hybrid_rise"]

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Ozellik sayisi:", len(features))
    print(
        "Macro-F1:",
        round(f1_score(actual, predictions, average="macro"), 4)
    )
    print(
        "Accuracy:",
        round(accuracy_score(actual, predictions), 4)
    )
    print(
        "Recall:",
        round(recall_score(actual, predictions), 4)
    )
    print(
        "Average Precision:",
        round(average_precision_score(actual, scores), 4)
    )
    print("Confusion Matrix:")
    print(confusion_matrix(actual, predictions, labels=[0, 1]))

    print("\nYUKSELEN KONULARDA GRUP BAZLI RECALL")

    for label, condition in [
        (
            "Gecmiste kuculen",
            validation["growth_3y_pct"] < 0,
        ),
        (
            "Gecmiste kuculmeyen",
            validation["growth_3y_pct"] >= 0,
        ),
    ]:
        mask = (
            (actual == 1) & condition
        ).to_numpy()

        total = int(mask.sum())
        correct = int(predictions[mask].sum())
        group_recall = correct / total if total else 0

        print(
            f"{label}: "
            f"{correct}/{total} "
            f"= %{group_recall * 100:.2f}"
        )


def main():
    df = pd.read_csv(DATA_PATH)

    train = df[df["split"] == "train"].copy()
    validation = df[
        df["split"] == "validation"
    ].copy()

    assert set(train["year"]) == {2020, 2021, 2022}
    assert set(validation["year"]) == {2023}

    features_10 = FEATURES.copy()

    features_9 = [
        feature for feature in FEATURES
        if feature != "growth_3y_pct"
    ]

    evaluate(
        train, validation, features_10,
        "MODEL A - 10 OZELLIK",
    )

    evaluate(
        train, validation, features_9,
        "MODEL B - 3 YILLIK BUYUME OLMADAN",
    )

    print("\n2024 test verisi KULLANILMADI.")
    print("Kaydedilen final model DEGISTIRILMEDI.")


if __name__ == "__main__":
    main()
