
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

OLD_FILE = (
    ROOT / "data/processed"
    / "work_details_120_topics_n30_cleaned.csv"
)

NEW_FILE = (
    ROOT / "data/processed"
    / "work_details_380_new_topics_n30_cleaned.csv"
)

OUTPUT_FILE = (
    ROOT / "data/processed"
    / "work_details_500_topics_n30_cleaned.csv"
)

OLD_TOPICS_FILE = (
    ROOT / "data/samples/topic_candidates_120.csv"
)

ALL_TOPICS_FILE = (
    ROOT / "data/samples/topic_candidates_500.csv"
)

csv.field_size_limit(10_000_000)


def read_topic_ids(path):
    with path.open(
        encoding="utf-8-sig", newline=""
    ) as file:
        return {
            row["topic_id"]
            for row in csv.DictReader(file)
        }


def main():
    old_ids = read_topic_ids(OLD_TOPICS_FILE)
    all_ids = read_topic_ids(ALL_TOPICS_FILE)
    new_ids = all_ids - old_ids

    if not (
        len(old_ids) == 120
        and len(all_ids) == 500
        and len(new_ids) == 380
    ):
        raise ValueError("Konu listeleri uyusmuyor.")

    seen_works = set()
    seen_topics = set()

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    temp_file = OUTPUT_FILE.with_suffix(".tmp")

    total = 0
    retracted_count = 0
    columns = None

    with temp_file.open(
        "w", encoding="utf-8-sig", newline=""
    ) as output:

        writer = None

        sources = [
            (OLD_FILE, old_ids, 28513),
            (NEW_FILE, new_ids, 90716),
        ]

        for source_path, allowed_ids, expected in sources:
            count = 0

            with source_path.open(
                encoding="utf-8-sig", newline=""
            ) as source:

                reader = csv.DictReader(source)

                if columns is None:
                    columns = reader.fieldnames

                    writer = csv.DictWriter(
                        output,
                        fieldnames=columns
                    )

                    writer.writeheader()

                elif set(reader.fieldnames) != set(columns):
                    raise ValueError(
                        "CSV sutun yapilari farkli."
                    )

                for row in reader:
                    work_id = row["work_id"]
                    topic_id = row["topic_id"]

                    if not work_id:
                        raise ValueError("Eksik work_id.")

                    if work_id in seen_works:
                        raise ValueError(
                            f"Tekrarlanan makale: {work_id}"
                        )

                    if topic_id not in allowed_ids:
                        raise ValueError(
                            f"Beklenmeyen konu: {topic_id}"
                        )

                    year = int(row["publication_year"])

                    if not 2018 <= year <= 2025:
                        raise ValueError(
                            f"Gecersiz yayin yili: {year}"
                        )

                    citation_history = json.loads(
                        row["counts_by_year_json"]
                    )

                    if not isinstance(citation_history, list):
                        raise ValueError(
                            f"Gecersiz atif gecmisi: {work_id}"
                        )

                    seen_works.add(work_id)
                    seen_topics.add(topic_id)

                    if row["is_retracted"].lower() == "true":
                        retracted_count += 1

                    writer.writerow(row)

                    count += 1
                    total += 1

            if count != expected:
                raise ValueError(
                    f"{source_path.name}: "
                    f"{expected} yerine {count} kayit."
                )

            print(
                f"{source_path.name}: {count} makale"
            )

    if total != 119229:
        raise ValueError("Toplam makale sayisi hatali.")

    if seen_topics != all_ids:
        raise ValueError(
            "Tum 500 konu veri setinde bulunmuyor."
        )

    if len(seen_works) != total:
        raise ValueError("Benzersiz makale kontrolu hatali.")

    temp_file.replace(OUTPUT_FILE)

    print("\n--- RESEARCHRADAR 500 TOPIC MERGE ---")
    print("Toplam makale:", total)
    print("Benzersiz makale:", len(seen_works))
    print("Benzersiz konu:", len(seen_topics))
    print("Geri cekilmis makale:", retracted_count)
    print("Tekrar eden makale: 0")

    print("\nBirlesik dosya:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
