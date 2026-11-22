
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

from final_v5_model import DATA_PATH, FEATURES

TARGET = "target_hybrid_rise"
THRESHOLD = 0.50


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

    model.fit(train[features], train[TARGET])

    scores = model.predict_proba(
        validation[features]
    )[:, 1]

    predictions = (scores >= THRESHOLD).astype(int)
    actual = validation[TARGET].to_numpy()

    cm = confusion_matrix(
        actual, predictions, labels=[0, 1]
    )

    negative_growth_mask = (
        (actual == 1)
        & (validation["growth_3y_pct"].to_numpy() < 0)
    )

    group_total = int(negative_growth_mask.sum())
    group_correct = int(
        predictions[negative_growth_mask].sum()
    )

    return {
        "Macro_F1": f1_score(
            actual, predictions, average="macro"
        ),
        "Accuracy": accuracy_score(actual, predictions),
        "Precision": precision_score(
            actual, predictions, zero_division=0
        ),
        "Recall": recall_score(
            actual, predictions, zero_division=0
        ),
        "AP": average_precision_score(actual, scores),
        "Negatif_3y_Recall": (
            group_correct / group_total
            if group_total else 0
        ),
        "Yanlis_Alarm": int(cm[0, 1]),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    # 2024 test verisini kullanma.
    df = df[df["year"].isin([
        2020, 2021, 2022, 2023
    ])].copy()

    # Bilim alanini sayisal sutunlara donustur.
    domain_columns = pd.get_dummies(
        df["domain"],
        prefix="domain",
        dtype=int,
    )

    df = pd.concat([df, domain_columns], axis=1)

    features_10 = FEATURES.copy()

    features_domain = (
        FEATURES + domain_columns.columns.tolist()
    )

    print("Domain sutunlari:")
    print(domain_columns.columns.tolist())

    print("\nMevcut ozellik sayisi:", len(features_10))
    print("Yeni ozellik sayisi:", len(features_domain))

    folds = [
        ([2020], 2021),
        ([2020, 2021], 2022),
        ([2020, 2021, 2022], 2023),
    ]

    results = []

    for train_years, val_year in folds:
        train = df[df["year"].isin(train_years)]
        validation = df[df["year"] == val_year]

        for name, features in [
            ("10 ozellik", features_10),
            ("Domain ekli", features_domain),
        ]:
            metrics = evaluate(
                train, validation, features
            )

            results.append({
                "Yil": val_year,
                "Model": name,
                **metrics,
            })

    result_df = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("YILLARA GORE MACRO-F1")
    print("=" * 65)

    print(
        result_df.pivot(
            index="Yil",
            columns="Model",
            values="Macro_F1",
        ).round(4).to_string()
    )

    print("\n" + "=" * 65)
    print("GECMISTE KUCULEN YUKSELENLERDE RECALL")
    print("=" * 65)

    print(
        result_df.pivot(
            index="Yil",
            columns="Model",
            values="Negatif_3y_Recall",
        ).round(4).to_string()
    )

    print("\n" + "=" * 65)
    print("TUM METRIKLER")
    print("=" * 65)

    print(result_df.round(4).to_string(index=False))

    print("\n2024 test verisi KULLANILMADI.")
    print("Kaydedilmis final model DEGISTIRILMEDI.")


if __name__ == "__main__":
    main()
