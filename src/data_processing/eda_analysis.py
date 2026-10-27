
from pathlib import Path
import pandas as pd

# Proje ana klasörü
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Mustafa'nın hazırladığı veri dosyaları
PILOT_FILE = (
    PROJECT_ROOT / "data" / "samples" / "T12072_pilot_300.csv"
)

YEARLY_FILE = (
    PROJECT_ROOT / "data" / "samples" / "T12072_yearly_counts.csv"
)

# Analiz çıktı klasörü
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def run_eda():

    # Dosyaları kontrol et
    if not PILOT_FILE.exists():
        raise FileNotFoundError(PILOT_FILE)

    if not YEARLY_FILE.exists():
        raise FileNotFoundError(YEARLY_FILE)

    # Verileri oku
    df = pd.read_csv(PILOT_FILE)
    yearly_df = pd.read_csv(YEARLY_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nResearchRadar - EDA Analizi")
    print("-" * 45)

    # 1. Yayın türlerinin dağılımı
    publication_types = (
        df["type"]
        .value_counts()
        .rename_axis("publication_type")
        .reset_index(name="count")
    )

    print("\n1. Yayın türleri:")
    print(publication_types.to_string(index=False))

    publication_types.to_csv(
        OUTPUT_DIR / "publication_types.csv",
        index=False
    )

    # 2. Atıf istatistikleri
    citations = pd.to_numeric(
        df["cited_by_count"],
        errors="coerce"
    )

    citation_statistics = citations.describe()

    print("\n2. Atıf istatistikleri:")
    print(citation_statistics)

    citation_statistics.to_csv(
        OUTPUT_DIR / "citation_statistics.csv",
        header=["value"]
    )

    zero_citation_count = int((citations == 0).sum())

    print(
        f"\nSıfır atıflı makale sayısı: {zero_citation_count}"
    )

    # 3. DOI eksikliğinin yıllara göre dağılımı
    doi_missing_by_year = (
        df.assign(doi_missing=df["doi"].isna())
        .groupby("publication_year")
        .agg(
            total_works=("work_id", "count"),
            missing_doi=("doi_missing", "sum")
        )
        .reset_index()
    )

    doi_missing_by_year["missing_percent"] = (
        doi_missing_by_year["missing_doi"]
        / doi_missing_by_year["total_works"]
        * 100
    ).round(2)

    print("\n3. DOI eksikliği - yıllara göre:")
    print(doi_missing_by_year.to_string(index=False))

    doi_missing_by_year.to_csv(
        OUTPUT_DIR / "doi_missing_by_year.csv",
        index=False
    )

    # 4. Gerçek yıllık yayın büyümesi
    yearly_df["publication_count"] = pd.to_numeric(
        yearly_df["publication_count"],
        errors="coerce"
    )

    # Eksik veya negatif yayın sayısını kabul etmiyoruz
    if (
        yearly_df["publication_count"].isna().any()
        or (yearly_df["publication_count"] < 0).any()
    ):
        raise ValueError("Geçersiz yıllık yayın sayısı var.")

    if yearly_df["year"].duplicated().any():
        raise ValueError("Aynı yıl birden fazla kez bulunuyor.")

    # Yılları kronolojik olarak sırala
    yearly_df = yearly_df.sort_values("year").copy()

    # Önceki yılın yayın sayısını al
    yearly_df["previous_year_count"] = (
        yearly_df["publication_count"].shift(1)
    )

    # Sıfıra bölmeyi önle
    previous = yearly_df["previous_year_count"]

    yearly_df["growth_percent"] = (
        (yearly_df["publication_count"] - previous)
        / previous.where(previous > 0)
        * 100
    ).round(2)

    # İlk yılın büyüme oranı hesaplanamaz (önceki yıl yok)
    print("\n4. Yıllık yayın büyümesi:")

    columns = [
        "year",
        "publication_count",
        "previous_year_count",
        "growth_percent"
    ]

    print(yearly_df[columns].to_string(index=False))

    # Büyüme analizini CSV olarak kaydet
    yearly_df[columns].to_csv(
        OUTPUT_DIR / "yearly_publication_growth.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nEDA raporları başarıyla oluşturuldu!")
    print(f"Rapor klasörü: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_eda()
