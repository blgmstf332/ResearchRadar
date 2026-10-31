
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEANED_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "work_details_120_topics_n30_cleaned.csv"
)

YEARLY_FILE = (
    PROJECT_ROOT / "data" / "raw"
    / "topic_year_counts_120.csv"
)

TOPICS_FILE = (
    PROJECT_ROOT / "data" / "raw"
    / "topic_candidates_120.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "eda_main"
PLOTS_DIR = OUTPUT_DIR / "plots"


def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / filename,
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(fig)


def run_eda():

    # 1. Veri dosyalarini oku
    works = pd.read_csv(CLEANED_FILE, low_memory=False)
    yearly = pd.read_csv(YEARLY_FILE)
    topics = pd.read_csv(TOPICS_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Konu kimliklerini dogrula
    if topics["topic_id"].duplicated().any():
        raise ValueError("Tekrarlanan konu kimligi var.")

    if works["work_id"].duplicated().any():
        raise ValueError("Tekrarlanan makale kimligi var.")

    # 3. Makalelere bilimsel alan bilgisini ekle
    topic_domains = topics[["topic_id", "domain"]]

    works = works.merge(
        topic_domains,
        on="topic_id",
        how="left",
        validate="many_to_one"
    )

    if works["domain"].isna().any():
        raise ValueError("Bilimsel alani bulunamayan makale var.")

    print("\nResearchRadar - Ana Veri EDA Analizi")
    print("-" * 45)
    print("Makale sayisi:", len(works))
    print("Konu sayisi:", topics["topic_id"].nunique())
    print("Bilimsel alan sayisi:", topics["domain"].nunique())

    # 4. Yayin turleri
    publication_types = (
        works["type"]
        .fillna("unknown")
        .value_counts()
        .rename_axis("publication_type")
        .reset_index(name="count")
    )

    publication_types.to_csv(
        OUTPUT_DIR / "publication_types_main.csv",
        index=False
    )

    print("\n1. Yayin turleri:")
    print(publication_types.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        publication_types["publication_type"],
        publication_types["count"]
    )
    ax.set_title("Yayin Turleri Dagilimi")
    ax.set_xlabel("Yayin turu")
    ax.set_ylabel("Orneklemdeki makale sayisi")
    save_plot(fig, "publication_types.png")

    # 5. Atif istatistikleri
    works["cited_by_count"] = pd.to_numeric(
        works["cited_by_count"],
        errors="coerce"
    )

    if (
        works["cited_by_count"].isna().any()
        or (works["cited_by_count"] < 0).any()
    ):
        raise ValueError("Gecersiz atif sayisi bulundu.")

    citation_stats = works["cited_by_count"].describe()

    citation_stats.to_csv(
        OUTPUT_DIR / "citation_statistics_main.csv",
        header=["value"]
    )

    print("\n2. Atif istatistikleri:")
    print(citation_stats)

    print(
        "Sifir atifli makale:",
        int((works["cited_by_count"] == 0).sum())
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        np.log1p(works["cited_by_count"]),
        bins=40
    )

    ax.set_title("Atif Dagilimi - Logaritmik Olcek")
    ax.set_xlabel("log(1 + atif sayisi)")
    ax.set_ylabel("Makale sayisi")

    save_plot(fig, "citation_distribution.png")

    # 6. Bilimsel alanlara gore konu sayilari
    domain_topics = (
        topics.groupby("domain")["topic_id"]
        .nunique()
        .reset_index(name="topic_count")
        .sort_values("topic_count", ascending=False)
    )

    domain_topics.to_csv(
        OUTPUT_DIR / "topics_by_domain.csv",
        index=False
    )

    print("\n3. Bilimsel alanlara gore konu sayisi:")
    print(domain_topics.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.barh(
        domain_topics["domain"],
        domain_topics["topic_count"]
    )

    ax.invert_yaxis()
    ax.set_title("Bilimsel Alanlara Gore Konu Dagilimi")
    ax.set_xlabel("Arastirma konusu sayisi")

    save_plot(fig, "topics_by_domain.png")

    # 7. Eksik DOI ve abstract analizi
    works["missing_doi"] = works["doi"].isna()
    works["missing_abstract"] = works["abstract"].isna()

    missing_by_domain = (
        works.groupby("domain")
        .agg(
            work_count=("work_id", "size"),
            missing_doi=("missing_doi", "sum"),
            missing_abstract=("missing_abstract", "sum")
        )
        .reset_index()
    )

    missing_by_domain["doi_percent"] = (
        missing_by_domain["missing_doi"]
        / missing_by_domain["work_count"] * 100
    ).round(2)

    missing_by_domain["abstract_percent"] = (
        missing_by_domain["missing_abstract"]
        / missing_by_domain["work_count"] * 100
    ).round(2)

    missing_by_domain.to_csv(
        OUTPUT_DIR / "missing_by_domain.csv",
        index=False
    )

    print("\n4. Bilimsel alanlara gore eksik veriler:")
    print(missing_by_domain.to_string(index=False))

    ax = (
        missing_by_domain
        .set_index("domain")[
            ["doi_percent", "abstract_percent"]
        ]
        .plot(kind="bar", figsize=(10, 6))
    )

    ax.set_title("Bilimsel Alanlara Gore Eksik Veri")
    ax.set_xlabel("Bilimsel alan")
    ax.set_ylabel("Eksik veri orani (%)")
    ax.legend(["DOI", "Abstract"])
    ax.tick_params(axis="x", rotation=20)

    save_plot(ax.figure, "missing_by_domain.png")

    # 8. Bilimsel alanlara gore atif istatistikleri
    citation_by_domain = (
        works.groupby("domain")["cited_by_count"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
    )

    citation_by_domain.to_csv(
        OUTPUT_DIR / "citations_by_domain.csv",
        index=False
    )

    print("\n5. Bilimsel alanlara gore atiflar:")
    print(citation_by_domain.to_string(index=False))

    # 9. Gercek konu-yil yayin sayilarini hazirla
    yearly["year"] = pd.to_numeric(
        yearly["year"], errors="coerce"
    )

    yearly["publication_count"] = pd.to_numeric(
        yearly["publication_count"], errors="coerce"
    )

    if (
        yearly["year"].isna().any()
        or (yearly["year"] % 1 != 0).any()
        or yearly["publication_count"].isna().any()
        or (yearly["publication_count"] < 0).any()
        or (yearly["publication_count"] % 1 != 0).any()
    ):
        raise ValueError("Gecersiz yillik yayin verisi.")

    if yearly.duplicated(["topic_id", "year"]).any():
        raise ValueError("Tekrarlanan konu-yil kaydi bulundu.")

    yearly = yearly.sort_values(
        ["topic_id", "year"]
    ).copy()

    # Onceki yil ile karsilastir
    grouped = yearly.groupby("topic_id")

    yearly["previous_count"] = (
        grouped["publication_count"].shift(1)
    )

    yearly["previous_year"] = grouped["year"].shift(1)

    valid_growth = (
        (yearly["previous_count"] > 0)
        & (yearly["year"] - yearly["previous_year"] == 1)
    )

    yearly["growth_percent"] = (
        (
            yearly["publication_count"]
            - yearly["previous_count"]
        )
        / yearly["previous_count"].where(valid_growth)
        * 100
    ).round(2)

    yearly.to_csv(
        OUTPUT_DIR / "topic_year_growth.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n6. Konu bazinda yillik buyume:")
    print(yearly[
        [
            "topic_id",
            "year",
            "publication_count",
            "growth_percent"
        ]
    ].head(12).to_string(index=False))

    # 10. Alan-yil bazinda konu yayin sayisi toplami
    # NOT: Bu toplamlarda farkli konular arasinda
    # ayni makale birden fazla kez sayilabilir.
    yearly_domain = (
        yearly.groupby(["year", "domain"])
        ["publication_count"]
        .sum()
        .reset_index(name="summed_topic_publications")
    )

    yearly_domain.to_csv(
        OUTPUT_DIR / "yearly_domain_topic_counts.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for domain, group in yearly_domain.groupby("domain"):
        ax.plot(
            group["year"],
            group["summed_topic_publications"],
            marker="o",
            label=domain
        )

    ax.set_title("Yillara Gore Alan Bazli Konu-Yayin Toplami")
    ax.set_xlabel("Yil")
    ax.set_ylabel("Toplanan konu-yayin sayisi")
    ax.legend(fontsize=8)

    save_plot(fig, "yearly_domain_publications.png")

    print("\nEDA analizi basariyla tamamlandi!")
    print("CSV raporlari:", OUTPUT_DIR)
    print("Grafikler:", PLOTS_DIR)


if __name__ == "__main__":
    run_eda()
