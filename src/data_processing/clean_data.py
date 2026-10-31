
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "raw"
    / "work_details_120_topics_n30.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "work_details_120_topics_n30_cleaned.csv"
)


def clean_data():
    # Orijinal dosyayi oku
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    original_count = len(df)

    # 1. Metin alanlarinin basindaki ve sonundaki
    # gereksiz bosluklari temizle
    text_columns = [
        "work_id",
        "topic_id",
        "topic_name",
        "title",
        "doi",
        "type",
        "abstract",
        "counts_by_year_json",
    ]

    for col in text_columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)

    # 2. Sayisal veri turlerini duzenle
    df["publication_year"] = pd.to_numeric(
        df["publication_year"], errors="raise"
    ).astype("Int64")

    df["cited_by_count"] = pd.to_numeric(
        df["cited_by_count"], errors="raise"
    ).astype("Int64")

    # 3. Tarih formatini standartlastir
    dates = pd.to_datetime(
        df["publication_date"], errors="raise"
    )

    df["publication_date"] = dates.dt.strftime(
        "%Y-%m-%d"
    )

    # 4. Boolean alanlarini standartlastir
    for col in ["has_abstract", "is_retracted"]:
        values = (
            df[col]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        if not values.isin(["true", "false"]).all():
            raise ValueError(
                f"Gecersiz boolean degeri: {col}"
            )

        df[col] = values.map({
            "true": True,
            "false": False
        }).astype("boolean")

    # 5. Kritik veri tutarliligini dogrula
    if df["work_id"].isna().any():
        raise ValueError("Eksik work_id bulundu.")

    if df["work_id"].duplicated().any():
        raise ValueError("Tekrarlanan work_id bulundu.")

    if not df["publication_year"].between(
        2018, 2025
    ).all():
        raise ValueError("Gecersiz yayin yili bulundu.")

    if (
        dates.dt.year != df["publication_year"]
    ).any():
        raise ValueError("Tarih ve yayin yili uyusmuyor.")

    if (
        df["cited_by_count"].isna().any()
        or (df["cited_by_count"] < 0).any()
    ):
        raise ValueError("Gecersiz atif sayisi bulundu.")

    if (
        df["has_abstract"] != df["abstract"].notna()
    ).any():
        raise ValueError("Abstract bayragi uyusmuyor.")

    # 6. Eksik veriler icin acik isaretler ekle
    df["missing_title"] = df["title"].isna()
    df["missing_doi"] = df["doi"].isna()
    df["missing_abstract"] = df["abstract"].isna()

    # 7. Hicbir makalenin kaybolmadigini dogrula
    if len(df) != original_count:
        raise ValueError("Kayit sayisi degisti!")

    # 8. Temizlenmis veriyi ayri dosyaya kaydet
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nResearchRadar - Veri Temizleme")
    print("-" * 40)
    print("Orijinal makale:", original_count)
    print("Temizlenmis makale:", len(df))
    print("Eksik baslik:", df["missing_title"].sum())
    print("Eksik DOI:", df["missing_doi"].sum())
    print("Eksik abstract:", df["missing_abstract"].sum())
    print(
        "Geri cekilmis makale:",
        df["is_retracted"].sum()
    )
    print("\nTemizlenmis CSV kaydedildi:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    clean_data()
