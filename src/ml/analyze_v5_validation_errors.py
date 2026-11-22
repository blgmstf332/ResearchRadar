
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

from final_v5_model import DATA_PATH, FEATURES, THRESHOLD


def main():
    df = pd.read_csv(DATA_PATH)

    # Yalnizca 2020-2022 ile egitiyoruz.
    train = df[df["split"] == "train"].copy()

    # Hatalari 2023 validation uzerinde inceliyoruz.
    validation = df[df["split"] == "validation"].copy()

    assert set(train["year"]) == {2020, 2021, 2022}
    assert set(validation["year"]) == {2023}

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        max_features=0.7,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(train[FEATURES], train["target_hybrid_rise"])

    validation["score"] = model.predict_proba(
        validation[FEATURES]
    )[:, 1]

    validation["prediction"] = (
        validation["score"] >= THRESHOLD
    ).astype(int)

    actual = validation["target_hybrid_rise"]
    predicted = validation["prediction"]

    validation["durum"] = "Dogru negatif (TN)"

    validation.loc[
        (actual == 1) & (predicted == 1), "durum"
    ] = "Dogru pozitif (TP)"

    validation.loc[
        (actual == 1) & (predicted == 0), "durum"
    ] = "Kacirilan yukselen (FN)"

    validation.loc[
        (actual == 0) & (predicted == 1), "durum"
    ] = "Yanlis alarm (FP)"

    print("=" * 65)
    print("V5 - 2023 VALIDATION HATA ANALIZI")
    print("=" * 65)

    print("Egitim kaydi:", len(train))
    print("Validation kaydi:", len(validation))
    print("Threshold:", THRESHOLD)

    print(
        "\nMacro-F1:",
        round(f1_score(actual, predicted, average="macro"), 4)
    )

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))

    print("\nTAHMIN DURUMLARI")
    print(validation["durum"].value_counts().to_string())

    analysis_features = [
        "publication_count",
        "growth_1y_pct",
        "growth_2y_pct",
        "growth_acceleration",
        "growth_stability",
        "citation_velocity_1y",
        "early_citation_mean",
        "growth_3y_pct",
    ]

    print("\nGRUPLARA GORE MEDYAN OZELLIKLER")
    print(
        validation.groupby("durum")[analysis_features]
        .median()
        .round(2)
        .to_string()
    )

    print("\nKACIRILAN YUKSELEN KONULAR - ILK 10")
    missed = validation[
        validation["durum"] == "Kacirilan yukselen (FN)"
    ].sort_values("score", ascending=False)

    print(
        missed[
            [
                "topic_name",
                "score",
                "growth_1y_pct",
                "growth_acceleration",
                "citation_velocity_1y",
            ]
        ].head(10).round(3).to_string(index=False)
    )

    print("\nYANLIS ALARMLAR - ILK 10")
    false_alarms = validation[
        validation["durum"] == "Yanlis alarm (FP)"
    ].sort_values("score", ascending=False)

    print(
        false_alarms[
            [
                "topic_name",
                "score",
                "growth_1y_pct",
                "growth_acceleration",
                "citation_velocity_1y",
            ]
        ].head(10).round(3).to_string(index=False)
    )

    print("\nOZELLIK ONEM SIRALAMASI")
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES,
    ).sort_values(ascending=False)

    print(importance.round(4).to_string())


    print("\n" + "=" * 65)
    print("GERCEKTEN YUKSELEN KONULARDA 3 YILLIK BUYUME")
    print("=" * 65)

    positives = validation[
        validation["target_hybrid_rise"] == 1
    ].copy()

    positives["buyume_grubu"] = positives[
        "growth_3y_pct"
    ].apply(
        lambda value: (
            "Gecmiste dusen"
            if value < 0
            else "Gecmiste dusmeyen"
        )
    )

    summary = positives.groupby("buyume_grubu").agg(
        konu_sayisi=("prediction", "size"),
        dogru_tahmin=("prediction", "sum"),
    )

    summary["recall"] = (
        summary["dogru_tahmin"]
        / summary["konu_sayisi"]
    ).round(4)

    print(summary.to_string())


    print("\n2024 test verisi kullanilmadi.")


if __name__ == "__main__":
    main()
