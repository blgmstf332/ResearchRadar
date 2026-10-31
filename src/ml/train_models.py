
import pandas as pd

# Ana ML veri setini oku
df = pd.read_csv("data/processed/ml_training_v3_hybrid.csv")

# Mustafa'nın hazırladığı veri ayrımını koru
train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# İlk deney: Atıf bilgileri olmadan model eğitimi
feature_cols = [
    "publication_count",
    "growth_1y_pct",
    "growth_2y_pct",
    "relative_growth_pp",
]

target_col = "target_hybrid_rise"

# Girdi özellikleri (X) ve hedef etiketler (y)
X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_val = val_df[feature_cols]
y_val = val_df[target_col]

print("\nKullanilan ozellikler:", feature_cols)
print("X_train boyutu:", X_train.shape)
print("X_val boyutu:", X_val.shape)
print("Eksik deger sayisi:", X_train.isna().sum().sum())


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Ilk model: Atif bilgileri olmadan Logistic Regression
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
)

# Modeli sadece train verisiyle egit
model.fit(X_train, y_train)

# Validation verisiyle tahmin yap
y_val_pred = model.predict(X_val)

print("\n--- Logistic Regression (Atifsiz) ---")
print("Validation Accuracy:", round(accuracy_score(y_val, y_val_pred), 4))
print("Validation Macro-F1:", round(f1_score(y_val, y_val_pred, average="macro"), 4))

from sklearn.metrics import average_precision_score

print("Logistic Regression Validation PR-AUC (AP):",
      round(average_precision_score(y_val, model.predict_proba(X_val)[:, 1]), 4))


from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

print("\n--- Detayli Model Sonuclari ---")

print("\nPrecision, Recall ve F1:")
print(classification_report(
    y_val,
    y_val_pred,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))



from sklearn.ensemble import RandomForestClassifier

# Ikinci model: Atif bilgileri olmadan Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Sadece train verisiyle egit
rf_model.fit(X_train, y_train)

# Validation verisinde tahmin yap
rf_val_pred = rf_model.predict(X_val)
from sklearn.metrics import average_precision_score

rf_val_prob = rf_model.predict_proba(X_val)[:, 1]
print("Random Forest Validation PR-AUC (AP):", round(average_precision_score(y_val, rf_val_prob), 4))

print("\n--- Random Forest (Atifsiz) ---")
print("Validation Accuracy:", round(accuracy_score(y_val, rf_val_pred), 4))
print("Validation Macro-F1:", round(f1_score(y_val, rf_val_pred, average="macro"), 4))

print("\nPrecision, Recall ve F1:")
print(classification_report(y_val, rf_val_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, rf_val_pred))



from lightgbm import LGBMClassifier

# Ucuncu model: Atif bilgileri olmadan LightGBM
lgbm_model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=10,
    class_weight="balanced",
    random_state=42,
    verbosity=-1
)

# Modeli sadece train verisiyle egit
lgbm_model.fit(X_train, y_train)

# Validation verisinde tahmin yap
lgbm_val_pred = lgbm_model.predict(X_val)

lgbm_val_prob = lgbm_model.predict_proba(X_val)[:, 1]
print("LightGBM Validation PR-AUC (AP):", round(average_precision_score(y_val, lgbm_val_prob), 4))

print("\n--- LightGBM (Atifsiz) ---")
print("Validation Accuracy:", round(accuracy_score(y_val, lgbm_val_pred), 4))
print("Validation Macro-F1:", round(f1_score(y_val, lgbm_val_pred, average="macro"), 4))

print("\nPrecision, Recall ve F1:")
print(classification_report(y_val, lgbm_val_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, lgbm_val_pred))



from sklearn.ensemble import StackingClassifier

# Hibrit model:
# Random Forest + LightGBM -> Logistic Regression

hybrid_model = StackingClassifier(
    estimators=[
        ("random_forest", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )),
        ("lightgbm", LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=10,
            class_weight="balanced",
            random_state=42,
            verbosity=-1
        ))
    ],
    final_estimator=LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),
    cv=5,
    stack_method="predict_proba",
    n_jobs=-1
)

# Hibrit modeli yalnizca train verisiyle egit
hybrid_model.fit(X_train, y_train)

# Validation verisinde tahmin yap
hybrid_val_pred = hybrid_model.predict(X_val)

print("\n--- HIBRIT MODEL (Atifsiz) ---")
print("Validation Accuracy:",
      round(accuracy_score(y_val, hybrid_val_pred), 4))
print("Validation Macro-F1:",
      round(f1_score(y_val, hybrid_val_pred, average="macro"), 4))

print("\nPrecision, Recall ve F1:")
print(classification_report(
    y_val, hybrid_val_pred, zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, hybrid_val_pred))
