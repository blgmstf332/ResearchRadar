
from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "work_details_120_topics_n30.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def analyze_data_quality(csv_path):
    # 1. Ana veri setini oku
    df = pd.read_csv(csv_path, low_memory=False)

    if df.empty:
        raise ValueError("Veri seti bos!")

    # 2. Bekledigimiz sutunlari kontrol et
    required_columns = [
        "work_id",
        "topic_id",
        "topic_name",
        "publication_year",
        "publication_date",
        "title",
        "doi",
        "type",
        "cited_by_count",
        "abstract",
        "has_abstract",
        "counts_by_year_json",
        "is_retracted",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Eksik sutunlar: {missing_columns}"
        )

    # 3. Bos metinleri eksik veri olarak kabul et
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace(
            r"^\s*$",
            pd.NA,
            regex=True
        )

    print("\nResearchRadar - Ana Veri Kalite Analizi")
    print("-" * 45)

    # 4. Eksik veri raporu
    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_percent": (
            df.isna().mean().values * 100
        ).round(2)
    })

    print("\n1. Eksik veri raporu:")
    print(report.to_string(index=False))

    # 5. Tekrar eden makale kontrolu
    duplicate_count = int(
        df["work_id"].dropna().duplicated().sum()
    )

    # 6. Yayin yili kontrolu
    years = pd.to_numeric(
        df["publication_year"],
        errors="coerce"
    )

    invalid_years = (
        df["publication_year"].notna()
        & (
            years.isna()
            | (years < 2018)
            | (years > 2025)
            | (years % 1 != 0)
        )
    )

    # 7. Yayin tarihi kontrolu
    dates = pd.to_datetime(
        df["publication_date"],
        errors="coerce"
    )

    invalid_dates = (
        df["publication_date"].notna()
        & dates.isna()
    )

    year_date_mismatch = (
        years.notna()
        & dates.notna()
        & (years != dates.dt.year)
    )

    # 8. Atif sayisi kontrolu
    citations = pd.to_numeric(
        df["cited_by_count"],
        errors="coerce"
    )

    invalid_citations = (
        df["cited_by_count"].notna()
        & (
            citations.isna()
            | (citations < 0)
            | (citations % 1 != 0)
        )
    )

    # 9. Abstract bayragi kontrolu
    abstract_flag = (
        df["has_abstract"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    valid_abstract_flags = abstract_flag.isin(
        ["true", "false"]
    )

    invalid_abstract_flags = (
        abstract_flag.notna()
        & ~valid_abstract_flags
    )

    actual_has_abstract = df["abstract"].notna()

    abstract_flag_mismatch = (
        (
            (abstract_flag == "true").fillna(False)
            != actual_has_abstract
        )
        & valid_abstract_flags
    )

    # 10. Geri cekilmis makaleler
    retracted_flag = (
        df["is_retracted"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    invalid_retracted_flags = (
        retracted_flag.notna()
        & ~retracted_flag.isin(["true", "false"])
    )

    retracted_count = int(
        (retracted_flag == "true").sum()
    )

    # 11. Yillik atif JSON kontrolu
    def is_invalid_json(value):
        if pd.isna(value):
            return False

        try:
            parsed = json.loads(str(value))
            return not isinstance(parsed, list)
        except (ValueError, TypeError):
            return True

    invalid_json_count = int(
        df["counts_by_year_json"]
        .apply(is_invalid_json)
        .sum()
    )

    # 12. Ozet kontrol tablosu
    summary = {
        "total_records": len(df),
        "unique_works": df["work_id"].nunique(),
        "unique_topics": df["topic_id"].nunique(),
        "missing_work_id": df["work_id"].isna().sum(),
        "missing_title": df["title"].isna().sum(),
        "missing_doi": df["doi"].isna().sum(),
        "missing_abstract": df["abstract"].isna().sum(),
        "duplicate_work_ids": duplicate_count,
        "invalid_years": invalid_years.sum(),
        "invalid_dates": invalid_dates.sum(),
        "year_date_mismatch": year_date_mismatch.sum(),
        "invalid_citations": invalid_citations.sum(),
        "invalid_abstract_flags": invalid_abstract_flags.sum(),
        "abstract_flag_mismatch": abstract_flag_mismatch.sum(),
        "invalid_retracted_flags": invalid_retracted_flags.sum(),
        "retracted_works": retracted_count,
        "invalid_citation_json": invalid_json_count,
    }

    summary_df = pd.DataFrame([
        {"check": key, "count": int(value)}
        for key, value in summary.items()
    ])

    print("\n2. Kalite kontrol ozeti:")
    print(summary_df.to_string(index=False))

    # 13. CSV raporlarini kaydet
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report.to_csv(
        OUTPUT_DIR / "data_quality_report_main.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary_df.to_csv(
        OUTPUT_DIR / "data_quality_summary_main.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nKalite raporlari basariyla kaydedildi!")
    print(f"Klasor: {OUTPUT_DIR}")

    return report


if __name__ == "__main__":
    if not INPUT_FILE.exists():
        print(f"Veri dosyasi bulunamadi: {INPUT_FILE}")
    else:
        analyze_data_quality(INPUT_FILE)
