
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# OpenAlex'in bildirdigi gercek yayin sayilari
full = pd.read_csv(
    ROOT / "data/samples/topic_year_counts_120.csv",
    dtype={"topic_id": str}
)

# Bizim indirdigimiz makaleler
works = pd.read_csv(
    ROOT / "data/processed/work_details_120_topics_n30.csv",
    dtype={"topic_id": str}
)

# Her konu ve yilda kac makale indirdik?
actual = (
    works.groupby(["topic_id", "publication_year"])
    .size()
    .reset_index(name="actual_sample")
    .rename(columns={"publication_year": "year"})
)

# Bir grupta 30'dan az yayin varsa hepsini bekleriz.
# 30 veya daha fazla varsa 30 ornek bekleriz.
full["expected_sample"] = full["publication_count"].clip(
    upper=30
).astype(int)

# Iki tabloyu karsilastir
result = full.merge(
    actual,
    on=["topic_id", "year"],
    how="left",
    validate="one_to_one"
)

result["actual_sample"] = (
    result["actual_sample"].fillna(0).astype(int)
)

result["difference"] = (
    result["expected_sample"] - result["actual_sample"]
)

mismatches = result[result["difference"] != 0]

print("\n--- RESEARCHRADAR ORNEKLEM DOGRULAMA ---")
print("Konu-yil kaydi:", len(result))
print("Beklenen toplam:", result["expected_sample"].sum())
print("Indirilen toplam:", result["actual_sample"].sum())
print("Tekrarlanan makale:", works["work_id"].duplicated().sum())
print("Uyusmayan grup sayisi:", len(mismatches))

if mismatches.empty:
    print("\nBASARILI: Tum gruplar beklenen sayida.")
else:
    print("\nINCELEME GEREKEN GRUPLAR:")
    print(
        mismatches[
            ["topic_id", "year", "publication_count",
             "expected_sample", "actual_sample", "difference"]
        ].to_string(index=False)
    )
