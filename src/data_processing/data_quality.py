
from pathlib import Path
import pandas as pd

# ResearchRadar projesinin ana klasörü
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def analyze_data_quality(csv_path):
    """
    OpenAlex makale verilerinin kalitesini kontrol eder.

    Kontroller:
    - Eksik değerler
    - Tekrarlanan makale kimlikleri
    - Geçersiz yayın yılları
    - Geçersiz atıf sayıları
    - Yayın tarihi ve yayın yılı tutarlılığı
    - Geri çekilmiş makaleler

    Orijinal veri setini değiştirmez.
    """

    # 1. CSV dosyasını oku
    df = pd.read_csv(csv_path)

    print("\nResearchRadar - Veri Kalite Raporu")
    print("-" * 45)

    print(f"Toplam kayıt: {len(df)}")
    print(f"Toplam sütun: {len(df.columns)}")

    if df.empty:
        raise ValueError("CSV dosyası boş.")

    # 2. Gerekli sütunları kontrol et
    required_columns = [
        "work_id",
        "title",
        "publication_year",
        "publication_date",
        "cited_by_count",
        "is_retracted"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"CSV dosyasında eksik sütunlar: {missing_columns}"
        )

    # 3. Eksik değerleri hesapla
    missing_values = df.isna().sum()

    print("\nEksik değer sayıları:")
    print(missing_values)

    # 4. Tekrarlanan makale kimliklerini kontrol et
    duplicate_count = (
        df["work_id"]
        .dropna()
        .duplicated()
        .sum()
    )

    print(f"\nTekrarlanan makale kaydı: {duplicate_count}")

    # 5. Yayın yıllarını kontrol et
    years = pd.to_numeric(
        df["publication_year"],
        errors="coerce"
    )

    invalid_years = (
        years.notna()
        & (
            (years < 2018)
            | (years > 2025)
            | (years % 1 != 0)
        )
    )

    print(f"Geçersiz yayın yılı: {invalid_years.sum()}")
    print(f"Boş/geçersiz yıl değeri: {years.isna().sum()}")

    # 6. Atıf sayılarını kontrol et
    citations = pd.to_numeric(
        df["cited_by_count"],
        errors="coerce"
    )

    invalid_citations = (
        citations.notna()
        & (
            (citations < 0)
            | (citations % 1 != 0)
        )
    )

    print(f"Geçersiz atıf sayısı: {invalid_citations.sum()}")
    print(f"Boş/geçersiz atıf değeri: {citations.isna().sum()}")

    # 7. Yayın tarihi kontrolleri
    dates = pd.to_datetime(
        df["publication_date"],
        errors="coerce"
    )

    missing_or_invalid_dates = dates.isna().sum()

    year_mismatches = (
        dates.notna()
        & years.notna()
        & (dates.dt.year != years)
    ).sum()

    print("\nYayın tarihi kontrolleri:")
    print(f"Eksik veya geçersiz tarih: {missing_or_invalid_dates}")
    print(f"Yayın yılı uyuşmazlığı: {year_mismatches}")

    if dates.notna().any():
        print(f"En eski yayın tarihi: {dates.min()}")
        print(f"En yeni yayın tarihi: {dates.max()}")

    # 8. Geri çekilmiş makaleleri kontrol et
    retracted = (
        df["is_retracted"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    retracted_count = (retracted == "true").sum()

    print(f"\nGeri çekilmiş makale sayısı: {retracted_count}")

    # 9. Sütun bazında kalite raporu
    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_values.values,
        "missing_percent": (
            df.isna().mean().values * 100
        ).round(2)
    })

    # 10. Rapor klasörünü oluştur
    output_dir = PROJECT_ROOT / "data" / "processed"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # 11. Kalite raporunu CSV dosyasına kaydet
    output_file = output_dir / "data_quality_report.csv"

    report.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nKalite raporu kaydedildi: {output_file}")

    return report


# Program doğrudan çalıştırıldığında
if __name__ == "__main__":

    # Mustafa'nın 300 makalelik pilot veri seti
    csv_file = (
        PROJECT_ROOT
        / "data"
        / "samples"
        / "T12072_pilot_300.csv"
    )

    if not csv_file.exists():
        print(f"Veri dosyası bulunamadı: {csv_file}")

    else:
        quality_report = analyze_data_quality(csv_file)

        print("\nSütun kalite raporu:")
        print(quality_report.to_string(index=False))
