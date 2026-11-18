
import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    ROOT / "data/raw/work_details_500_expansion/sample_30"
)

OLD_TOPICS = ROOT / "data/samples/topic_candidates_120.csv"
ALL_TOPICS = ROOT / "data/samples/topic_candidates_500.csv"

OUTPUT_FILE = (
    ROOT / "data/raw/work_details_380_new_topics_n30.csv"
)

COLUMNS = [
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


def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return None

    words = []

    for word, positions in inverted_index.items():
        for position in positions:
            words.append((position, word))

    words.sort(key=lambda item: item[0])

    return " ".join(word for _, word in words)


def main():
    old_df = pd.read_csv(
        OLD_TOPICS, dtype={"topic_id": str}
    )

    all_df = pd.read_csv(
        ALL_TOPICS, dtype={"topic_id": str}
    )

    old_ids = set(old_df["topic_id"])

    new_df = all_df[
        ~all_df["topic_id"].isin(old_ids)
    ].copy()

    if len(old_ids) != 120 or len(new_df) != 380:
        raise ValueError("120 + 380 konu kontrolu basarisiz.")

    if new_df["topic_id"].duplicated().any():
        raise ValueError("Tekrarlanan konu ID bulundu.")

    if not RAW_DIR.exists():
        raise FileNotFoundError(RAW_DIR)

    # 380 konu x 8 yil = 3040 JSONL dosyasi
    expected_files = {
        f"{topic_id}_{year}.jsonl"
        for topic_id in new_df["topic_id"]
        for year in range(2018, 2026)
    }

    actual_files = {
        path.name for path in RAW_DIR.glob("*.jsonl")
    }

    missing_files = expected_files - actual_files
    extra_files = actual_files - expected_files

    if missing_files or extra_files:
        raise ValueError(
            f"Eksik dosya: {len(missing_files)}, "
            f"fazla dosya: {len(extra_files)}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    temporary = OUTPUT_FILE.with_suffix(".tmp")

    seen_ids = set()
    total = 0
    abstract_count = 0
    retracted_count = 0

    with temporary.open(
        "w", encoding="utf-8-sig", newline=""
    ) as output:

        writer = csv.DictWriter(
            output, fieldnames=COLUMNS
        )
        writer.writeheader()

        for index, topic in enumerate(
            new_df.itertuples(index=False), start=1
        ):
            topic_total = 0

            for year in range(2018, 2026):
                file_path = (
                    RAW_DIR / f"{topic.topic_id}_{year}.jsonl"
                )

                year_count = 0

                with file_path.open(
                    encoding="utf-8"
                ) as source:

                    for line in source:
                        if not line.strip():
                            continue

                        work = json.loads(line)

                        work_id = work.get("id")

                        if not work_id:
                            raise ValueError(
                                f"Eksik work_id: {file_path}"
                            )

                        if work_id in seen_ids:
                            raise ValueError(
                                f"Tekrarlanan makale: {work_id}"
                            )

                        if work.get("publication_year") != year:
                            raise ValueError(
                                f"Yayin yili hatasi: {work_id}"
                            )

                        primary = (
                            work.get("primary_topic") or {}
                        )

                        actual_topic = (
                            str(primary.get("id") or "")
                            .rstrip("/")
                            .split("/")[-1]
                        )

                        if actual_topic != topic.topic_id:
                            raise ValueError(
                                f"Konu uyusmazligi: {work_id}"
                            )

                        abstract = reconstruct_abstract(
                            work.get("abstract_inverted_index")
                        )

                        is_retracted = work.get("is_retracted")

                        if not isinstance(is_retracted, bool):
                            raise ValueError(
                                f"Gecersiz is_retracted: {work_id}"
                            )

                        writer.writerow({
                            "work_id": work_id,
                            "topic_id": topic.topic_id,
                            "topic_name": topic.topic_name,
                            "publication_year": year,
                            "publication_date": work.get(
                                "publication_date"
                            ),
                            "title": work.get("title"),
                            "doi": work.get("doi"),
                            "type": work.get("type"),
                            "cited_by_count": work.get(
                                "cited_by_count"
                            ),
                            "abstract": abstract,
                            "has_abstract": bool(abstract),
                            "counts_by_year_json": json.dumps(
                                work.get("counts_by_year") or [],
                                ensure_ascii=False
                            ),
                            "is_retracted": is_retracted,
                        })

                        seen_ids.add(work_id)
                        total += 1
                        topic_total += 1
                        year_count += 1

                        if abstract:
                            abstract_count += 1

                        if is_retracted:
                            retracted_count += 1

                if year_count > 30:
                    raise ValueError(
                        f"30'dan fazla makale: {file_path}"
                    )

            if index % 25 == 0 or index == 380:
                print(
                    f"[{index}/380] "
                    f"Islenen makale: {total}"
                )

    if total != 90716:
        raise ValueError(
            f"90716 makale bekleniyor, bulunan: {total}"
        )

    temporary.replace(OUTPUT_FILE)

    print("\n--- RESEARCHRADAR CSV DONUSUMU ---")
    print("Islenen konu:", len(new_df))
    print("JSONL dosyasi:", len(expected_files))
    print("Makale kaydi:", total)
    print("Benzersiz makale:", len(seen_ids))
    print("Abstract bulunan:", abstract_count)
    print("Abstract bulunmayan:", total - abstract_count)
    print("Geri cekilmis makale:", retracted_count)
    print("\nCSV dosyasi:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
