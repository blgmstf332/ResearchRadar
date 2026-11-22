
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    recall_score,
    average_precision_score,
)

from final_v5_model import DATA_PATH, FEATURES, THRESHOLD

TARGET = "target_hybrid_rise"


def evaluate(train, validation, extra_weight):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        max_features=0.7,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Normalde tum orneklerin agirligi 1.
    weights = np.ones(len(train))

    # Gecmis buyumesi negatif ama gercekte
    # yukselen konulara daha fazla onem ver.
    recovery_mask = (
        (train[TARGET] == 1)
        & (train["growth_3y_pct"] < 0)
    )

    weights[recovery_mask.to_numpy()] = extra_weight

    model.fit(
        train[FEATURES],
        train[TARGET],
        sample_weight=weights,
    )

    scores = model.predict_proba(
        validation[FEATURES]
    )[:, 1]

    predictions = (scores >= THRESHOLD).astype(int)
    actual = validation[TARGET]

    # Ozellikle kacirdigimiz grubu incele.
    group_mask = (
        (actual == 1)
        & (validation["growth_3y_pct"] < 0)
    ).to_numpy()

    group_total = int(group_mask.sum())
    group_correct = int(predictions[group_mask].sum())

    return {
        "Macro-F1": f1_score(
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
            if group_total > 0 else 0
        ),
        "Yanlis_Alarm": int(
            ((actual.to_numpy() == 0) & (predictions == 1)).sum()
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    # 2024 test verisi kesinlikle dahil edilmiyor.
    df = df[
        df["year"].isin([2020, 2021, 2022, 2023])
    ].copy()

    folds = [
        ([2020], 2021),
        ([2020, 2021], 2022),
        ([2020, 2021, 2022], 2023),
    ]

    results = []

    for train_years, validation_year in folds:
        train = df[
            df["year"].isin(train_years)
        ].copy()

        validation = df[
            df["year"] == validation_year
        ].copy()

        print("\n" + "=" * 65)
        print(
            f"Egitim: {train_years} | "
            f"Dogrulama: {validation_year}"
        )
        print("=" * 65)

        for name, weight in [
            ("Normal RF", 1.0),
            ("Agirlikli RF", 2.0),
        ]:
            metrics = evaluate(
                train, validation, weight
            )

            print(f"\n{name}")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")

            results.append({
                "Yil": validation_year,
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
            values="Macro-F1",
        ).round(4).to_string()
    )

    print("\n" + "=" * 65)
    print("GECMIS BUYUMESI NEGATIF KONULARDA RECALL")
    print("=" * 65)

    print(
        result_df.pivot(
            index="Yil",
            columns="Model",
            values="Negatif_3y_Recall",
        ).round(4).to_string()
    )

    print("\n2024 test verisi KULLANILMADI.")
    print("Final model DEGISTIRILMEDI.")


if __name__ == "__main__":
    main()
