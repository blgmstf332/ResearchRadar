
import pandas as pd

# Ana veri setini oku
df = pd.read_csv("data/processed/ml_training_v3_hybrid.csv")

# Atif bilgileri olmadan kullanacagimiz ozellikler
feature_cols = [
    "publication_count",
    "growth_1y_pct",
    "growth_2y_pct",
    "relative_growth_pp",
]

target_col = "target_hybrid_rise"

# Yillara gore egitim verilerini ayir
train_2020 = df[(df["split"] == "train") & (df["year"] == 2020)].copy()
train_2021 = df[(df["split"] == "train") & (df["year"] == 2021)].copy()
train_2022 = df[(df["split"] == "train") & (df["year"] == 2022)].copy()

# 2023: model secimi, 2024: nihai test
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

print("2020 egitim:", len(train_2020))
print("2021 egitim:", len(train_2021))
print("2022 egitim:", len(train_2022))
print("Validation:", len(val_df))
print("Test:", len(test_df))



from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# 2020 yilinin ozellikleri ve hedef etiketleri
X_2020 = train_2020[feature_cols]
y_2020 = train_2020[target_col]

# Random Forest
rf_2020 = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# LightGBM
lgbm_2020 = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=10,
    class_weight="balanced",
    random_state=42,
    verbosity=-1
)

# Iki modeli de yalnizca 2020 verisiyle egit
rf_2020.fit(X_2020, y_2020)
lgbm_2020.fit(X_2020, y_2020)

print("\n2020 verisiyle iki model egitildi!")
print("Random Forest:", rf_2020.__class__.__name__)
print("LightGBM:", lgbm_2020.__class__.__name__)



# 2021 yilinin ozelliklerini hazirla
X_2021 = train_2021[feature_cols]

# 2020 ile egitilen modellerden 2021 tahminleri al
rf_prob_2021 = rf_2020.predict_proba(X_2021)[:, 1]
lgbm_prob_2021 = lgbm_2020.predict_proba(X_2021)[:, 1]

# Iki modelin tahminlerini bir tabloda birlestir
meta_2021 = pd.DataFrame({
    "rf_probability": rf_prob_2021,
    "lgbm_probability": lgbm_prob_2021,
    "target": train_2021[target_col].to_numpy()
})

print("\n--- 2021 Meta Verileri ---")
print(meta_2021.head())
print("Toplam kayit:", len(meta_2021))



# 2020 ve 2021 verilerini birlestir
train_until_2021 = pd.concat(
    [train_2020, train_2021],
    ignore_index=True
)

X_until_2021 = train_until_2021[feature_cols]
y_until_2021 = train_until_2021[target_col]

# Yeni modeller olustur
rf_2022 = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

lgbm_2022 = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=10,
    class_weight="balanced",
    random_state=42,
    verbosity=-1
)

# Yalnizca 2020 ve 2021 verileriyle egit
rf_2022.fit(X_until_2021, y_until_2021)
lgbm_2022.fit(X_until_2021, y_until_2021)

# 2022 verileri icin tahmin olasiliklari
X_2022 = train_2022[feature_cols]

rf_prob_2022 = rf_2022.predict_proba(X_2022)[:, 1]
lgbm_prob_2022 = lgbm_2022.predict_proba(X_2022)[:, 1]

# 2022 meta verilerini olustur
meta_2022 = pd.DataFrame({
    "rf_probability": rf_prob_2022,
    "lgbm_probability": lgbm_prob_2022,
    "target": train_2022[target_col].to_numpy()
})

print("\n--- 2022 Meta Verileri ---")
print(meta_2022.head())
print("Toplam kayit:", len(meta_2022))



from sklearn.linear_model import LogisticRegression

# 2021 ve 2022 meta verilerini birlestir
meta_train = pd.concat(
    [meta_2021, meta_2022],
    ignore_index=True
)

# Temel modellerin tahminleri
meta_features = [
    "rf_probability",
    "lgbm_probability"
]

X_meta = meta_train[meta_features]
y_meta = meta_train["target"]

# Hibrit modelin son asamasi: Logistic Regression
meta_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

meta_model.fit(X_meta, y_meta)

print("\n--- HIBRIT META MODEL ---")
print("Meta egitim kaydi:", len(meta_train))
print("Meta ozellikleri:", meta_features)
print("Logistic Regression egitildi!")



# 2020, 2021 ve 2022 verilerini birlestir
full_train = pd.concat(
    [train_2020, train_2021, train_2022],
    ignore_index=True
)

X_full_train = full_train[feature_cols]
y_full_train = full_train[target_col]

# Tum gecmis verilerle egitilecek temel modeller
rf_final = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

lgbm_final = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=10,
    class_weight="balanced",
    random_state=42,
    verbosity=-1
)

rf_final.fit(X_full_train, y_full_train)
lgbm_final.fit(X_full_train, y_full_train)

print("\n--- TEMEL MODELLER HAZIR ---")
print("Toplam egitim kaydi:", len(full_train))
print("Random Forest ve LightGBM egitildi!")



from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 2023 validation verileri
X_val = val_df[feature_cols]
y_val = val_df[target_col]

# Random Forest ve LightGBM olasiliklari
rf_val_prob = rf_final.predict_proba(X_val)[:, 1]
lgbm_val_prob = lgbm_final.predict_proba(X_val)[:, 1]

# Logistic Regression icin meta ozellikler
meta_val = pd.DataFrame({
    "rf_probability": rf_val_prob,
    "lgbm_probability": lgbm_val_prob
})

# Hibrit modelin son tahminleri
hybrid_val_pred = meta_model.predict(meta_val)

print("\n--- ZAMAN UYUMLU HIBRIT MODEL ---")
print("Validation Accuracy:",
      round(accuracy_score(y_val, hybrid_val_pred), 4))
print("Validation Precision:",
      round(precision_score(y_val, hybrid_val_pred, zero_division=0), 4))
print("Validation Recall:",
      round(recall_score(y_val, hybrid_val_pred, zero_division=0), 4))
print("Validation Macro-F1:",
      round(f1_score(y_val, hybrid_val_pred, average="macro"), 4))

print("\nSiniflandirma Raporu:")
print(classification_report(y_val, hybrid_val_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, hybrid_val_pred))


from sklearn.metrics import average_precision_score

# Hibrit modelin 2023 icin yukselis olasiliklari
hybrid_val_prob = meta_model.predict_proba(meta_val)[:, 1]

print("\nValidation Average Precision (PR-AUC):",
      round(average_precision_score(y_val, hybrid_val_prob), 4))
