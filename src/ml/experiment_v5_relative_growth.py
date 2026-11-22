
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
)

from final_v5_model import DATA_PATH, FEATURES, THRESHOLD


def evaluate_model(train, validation, features, model_name):
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

    print("\n" + "=" * 55)
    print(model_name)
    print("=" * 55)

    print(
        "Accuracy:",
        round(accuracy_score(y_true, predictions), 4)
    )
    print(
        "Macro-F1:",
        round(f1_score(y_true, predictions, average="macro"), 4)
    )
    print(
        "Precision (sinif 1):",
        round(precision_score(y_true, predictions, zero_division=0), 4)
    )
    print(
        "Recall (sinif 1):",
        round(recall_score(y_true, predictions, zero_division=0), 4)
    )
    print(
        "Average Precision:",
        round(average_precision_score(y_true, scores), 4)
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, predictions, labels=[0, 1]))

    return f1_score(y_true, predictions, average="macro")


def main():
    df = pd.read_csv(DATA_PATH)

    # 2024 test verisini tamamen disarida birak.
    df = df[
        df["split"].isin(["train", "validation"])
    ].copy()

    # Her yil ve alan icin buyume medyanini hesapla.
    df["domain_median_growth"] = df.groupby(
        ["year", "domain"]
    )["growth_1y_pct"].transform("median")

    # Yeni ozellik: alanina gore goreli buyume.
    df["relative_growth_pp"] = (
        df["growth_1y_pct"] - df["domain_median_growth"]
    )

    train = df[df["split"] == "train"].copy()
    validation = df[df["split"] == "validation"].copy()

    assert set(train["year"]) == {2020, 2021, 2022}
    assert set(validation["year"]) == {2023}

    old_features = FEATURES.copy()

    new_features = FEATURES + ["relative_growth_pp"]

    print("Egitim kaydi:", len(train))
    print("Validation kaydi:", len(validation))
    print("Karar esigi:", THRESHOLD)

    old_f1 = evaluate_model(
        train,
        validation,
        old_features,
        "MEVCUT MODEL - 10 OZELLIK",
    )

    new_f1 = evaluate_model(
        train,
        validation,
        new_features,
        "YENI MODEL - 11 OZELLIK",
    )

    print("\n" + "=" * 55)
    print("KARSILASTIRMA")
    print("=" * 55)

    print("Eski Macro-F1:", round(old_f1, 4))
    print("Yeni Macro-F1:", round(new_f1, 4))
    print("Fark:", round(new_f1 - old_f1, 4))

    print("\n2024 test verisi KULLANILMADI.")
    print("Kaydedilmis final model DEGISTIRILMEDI.")


if __name__ == "__main__":
    main()
