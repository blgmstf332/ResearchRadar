from pathlib import Path
import pandas as pd

# ResearchRadar projesinin ana klasörü
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def analyze_data_quality(csv_path):
    """
    OpenAlex makale verilerinin kalitesini analiz eder.
    Verileri değiştirmez, yalnızca sorunları raporlar.
    """

    # CSV dosyasını oku
    df = pd.read_csv(csv_path)

    print("\nResearchRadar - Veri Kalite Raporu")
    print("-" * 40)

    print(f"Toplam kayıt: {len(df)}")
    print(f"Toplam sütun: {len(df.columns)}")

    if df.empty:
        raise ValueError("CSV dosyası boş.")

    required_columns = [
        "work_id",
        "title",
        "publication_year",
        "cited_by_count"
    ]

    # Gerekli sütunlar var mı?
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"CSV dosyasında eksik sütunlar: {missing_columns}"
        )

    # Eksik değer analizi
    print("\nEksik değer sayıları:")
    print(df.isna().sum())

    # Makale kimliğine göre tekrar kontrolü
    duplicate_count = df["work_id"].dropna().duplicated().sum()

    print(f"\nTekrarlanan makale kaydı: {duplicate_count}")

    # Yayın yıllarını sayısal değere dönüştür
    years = pd.to_numeric(
        df["publication_year"],
        errors="coerce"
    )

    invalid_years = (
        years.notna()
        & ((years < 1900) | (years > 2026) | (years % 1 != 0))
    )

    print(f"Geçersiz yayın yılı: {invalid_years.sum()}")

    # Atıf kontrolü
    citations = pd.to_numeric(
        df["cited_by_count"],
        errors="coerce"
    )

    invalid_citations = (
        citations.notna()
        & ((citations < 0) | (citations % 1 != 0))
    )

    print(f"Geçersiz atıf sayısı: {invalid_citations.sum()}")

    # Eksik ve hatalı sayısal değerleri ayrıca raporla
    print(f"Boş/geçersiz yıl değeri: {years.isna().sum()}")
    print(f"Boş/geçersiz atıf değeri: {citations.isna().sum()}")

    # Rapor tablosu
    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_percent": (
            df.isna().mean().values * 100
        ).round(2)
    })

    return report


if __name__ == "__main__":

    csv_file = (
        PROJECT_ROOT / "data" / "samples" / "pilot_10.csv"
    )

    if not csv_file.exists():
        print(f"Örnek veri dosyası henüz bulunamadı: {csv_file}")
        print("Mustafa'nın pilot CSV verisini bekliyoruz.")

    else:
        quality_report = analyze_data_quality(csv_file)

        print("\nSütun kalite raporu:")
        print(quality_report.to_string(index=False))