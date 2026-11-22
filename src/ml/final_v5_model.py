
from pathlib import Path
import math

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data" / "processed"
    / "ml_training_v5_500_citations.csv"
)

MODEL_PATH = ROOT / "models" / "v5_random_forest.joblib"

FEATURES = [
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

TARGET = "target_hybrid_rise"
THRESHOLD = 0.50


def train_and_save():
    df = pd.read_csv(DATA_PATH)

    # 2020-2023 kullanilir.
    # 2024 test verisi egitime ALINMAZ.
    training_df = df[
        df["split"].isin(["train", "validation"])
    ].copy()

    assert set(training_df["year"]) == {
        2020, 2021, 2022, 2023
    }

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
        training_df[FEATURES],
        training_df[TARGET],
    )

    # Model, ozellik sirasi ve threshold birlikte saklanir.
    bundle = {
        "model": model,
        "features": FEATURES,
        "threshold": THRESHOLD,
        "training_years": [2020, 2021, 2022, 2023],
        "model_name": "Random Forest V5",
    }

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(bundle, MODEL_PATH)

    print("MODEL BASARIYLA KAYDEDILDI")
    print("Model yolu:", MODEL_PATH)
    print("Egitim kaydi:", len(training_df))
    print("Ozellik sayisi:", len(FEATURES))
    print("Threshold:", THRESHOLD)
    print("2024 test verisi KULLANILMADI.")


def predict_v5(feature_values: dict) -> dict:
    """
    Tek bir konu icin tahmin uretir.

    feature_values:
        10 sayisal ozelligi iceren sozluk.

    Donus:
        prediction: 0 veya 1
        probability: yukselme olasiligi
        threshold: karar esigi
    """

    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    features = bundle["features"]
    threshold = bundle["threshold"]

    missing = [
        feature for feature in features
        if feature not in feature_values
    ]

    if missing:
        raise ValueError(
            f"Eksik ozellikler: {missing}"
        )

    values = {}

    for feature in features:
        try:
            value = float(feature_values[feature])
        except (TypeError, ValueError):
            raise ValueError(
                f"Gecersiz sayisal deger: {feature}"
            )

        if not math.isfinite(value):
            raise ValueError(
                f"Sonlu olmayan deger: {feature}"
            )

        values[feature] = value

    # Sutun sirasi egitimdekiyle ayni olmali.
    X = pd.DataFrame(
        [values],
        columns=features,
    )

    probability = float(
        model.predict_proba(X)[0, 1]
    )

    prediction = int(
        probability >= threshold
    )

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "threshold": threshold,
    }


if __name__ == "__main__":
    train_and_save()
