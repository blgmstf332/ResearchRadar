
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
)

from final_v5_model import DATA_PATH, FEATURES

TARGET = "target_hybrid_rise"
THRESHOLD = 0.50


def evaluate(train, val, features):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        max_features=0.7,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(train[features], train[TARGET])

    scores = model.predict_proba(val[features])[:, 1]
    predictions = (scores >= THRESHOLD).astype(int)
    actual = val[TARGET].to_numpy()

    # Gecmiste kuculmus ama gercekte yukselen konular
    group_mask = (
        (actual == 1)
        & (val["growth_3y_pct"].to_numpy() < 0)
    )

    group_total = int(group_mask.sum())
    group_correct = int(predictions[group_mask].sum())

    cm = confusion_matrix(
        actual, predictions, labels=[0, 1]
    )

    return {
        "Macro_F1": f1_score(
            actual, predictions, average="macro"
        ),
        "Recall": recall_score(
            actual, predictions, zero_division=0
        ),
        "AP": average_precision_score(
            actual, scores
        ),
        "Negatif_3y_Recall": (
            group_correct / group_total
            if group_total else 0
        ),
        "Yanlis_Alarm": int(cm[0, 1]),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    # 2024 final test verisini kullanma.
    df = df[df["year"].between(2020, 2023)].copy()

    # Yeni toparlanma ozellikleri
    df["rebound_gap"] = (
        df["growth_1y_pct"] - df["growth_3y_pct"]
    )

    df["rebound_flag"] = (
        (df["growth_3y_pct"] < 0)
        & (df["growth_1y_pct"] > 0)
    ).astype(int)

    old_features = FEATURES.copy()
    new_features = FEATURES + [
        "rebound_gap",
        "rebound_flag",
    ]

    folds = [
        ([2020], 2021),
        ([2020, 2021], 2022),
        ([2020, 2021, 2022], 2023),
    ]

    results = []

    for train_years, val_year in folds:
        train = df[df["year"].isin(train_years)]
        val = df[df["year"] == val_year]

        for name, features in [
            ("10 ozellik", old_features),
            ("12 ozellik", new_features),
        ]:
            metrics = evaluate(
                train, val, features
            )

            results.append({
                "Yil": val_year,
                "Model": name,
                **metrics,
            })

    result_df = pd.DataFrame(results)

    print("=" * 65)
    print("V5 - TOPARLANMA OZELLIKLERI DENEYI")
    print("=" * 65)

    print("\nYILLARA GORE MACRO-F1")
    print(
        result_df.pivot(
            index="Yil",
            columns="Model",
            values="Macro_F1",
        ).round(4).to_string()
    )

    print("\nGECMISTE KUCULEN YUKSELENLERDE RECALL")
    print(
        result_df.pivot(
            index="Yil",
            columns="Model",
            values="Negatif_3y_Recall",
        ).round(4).to_string()
    )

    print("\nDIGER METRIKLER")
    print(result_df.round(4).to_string(index=False))

    print("\n2024 test verisi KULLANILMADI.")
    print("Kaydedilmis final model DEGISTIRILMEDI.")


if __name__ == "__main__":
    main()
