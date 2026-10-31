
from pathlib import Path

# Onceki zaman uyumlu hibrit model kodunu oku
source_path = Path("src/ml/train_temporal_stacking.py")
source_code = source_path.read_text(encoding="utf-8")

# Atif ozelliklerini ekle
old_features = '''feature_cols = [
    "publication_count",
    "growth_1y_pct",
    "growth_2y_pct",
    "relative_growth_pp",
]'''

new_features = '''feature_cols = [
    "publication_count",
    "growth_1y_pct",
    "growth_2y_pct",
    "relative_growth_pp",
    "citation_velocity_1y",
    "citation_velocity_median",
    "early_citation_mean",
    "citation_sample_size",
]'''

if old_features not in source_code:
    raise ValueError("Ozellik listesi bulunamadi. Kod degistirilmedi.")

# Onceki kodu atifli ozelliklerle calistir
updated_code = source_code.replace(old_features, new_features, 1)

print("ATIFLI HIBRIT MODEL CALISIYOR...")
exec(compile(updated_code, str(source_path), "exec"))
