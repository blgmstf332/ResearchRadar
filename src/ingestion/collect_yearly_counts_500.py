
import argparse
import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("OPENALEX_API_KEY")

TOPICS_FILE = ROOT / "data/samples/topic_candidates_500.csv"
OUTPUT_FILE = ROOT / "data/samples/topic_year_counts_500.csv"
PROGRESS_FILE = (
    ROOT / "data/processed/topic_year_counts_500_progress.csv"
)

API_URL = "https://api.openalex.org/works"

START_YEAR = 2016
END_YEAR = 2025
YEARS = set(range(START_YEAR, END_YEAR + 1))

COLUMNS = [
    "topic_id",
    "topic_name",
    "domain",
    "field",
    "subfield",
    "year",
    "publication_count",
    "retrieved_at_utc",
]


def read_csv(path):
    if not path.exists():
        return []

    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    # Gecici dosyaya yazip sonra degistiriyoruz.
    temporary = path.with_suffix(".tmp")

    with open(
        temporary,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)


def fetch_year_counts(session, topic_id):
    params = {
        "api_key": API_KEY,
        "filter": (
            f"primary_topic.id:{topic_id},"
            f"publication_year:{START_YEAR}-{END_YEAR},"
            "type:article|preprint"
        ),
        "group_by": "publication_year",
        "per_page": 100,
    }

    for attempt in range(5):
        try:
            response = session.get(
                API_URL,
                params=params,
                timeout=45
            )
        except requests.RequestException:
            if attempt == 4:
                raise RuntimeError(
                    f"{topic_id} icin baglanti basarisiz."
                )
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 429:
            reason = response.json().get("reason", "")

            if reason == "daily_budget":
                raise RuntimeError(
                    "OpenAlex gunluk API kotasi doldu."
                )

            wait = response.headers.get("Retry-After")
            try:
                seconds = max(float(wait), 1)
            except (ValueError, TypeError):
                seconds = 2 ** attempt

            time.sleep(seconds)
            continue

        if response.status_code in (500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue

        if response.status_code != 200:
            raise RuntimeError(
                f"{topic_id}: HTTP {response.status_code}"
            )

        data = response.json()

        counts = {year: 0 for year in YEARS}

        for item in data.get("group_by", []):
            year = int(item["key"])

            if year not in YEARS:
                raise ValueError(
                    f"Beklenmeyen yil: {year}"
                )

            counts[year] += int(item["count"])

        # Filtrelenen tum kayitlarin gruplarda
        # temsil edilip edilmedigini kontrol et.
        api_total = int(data["meta"]["count"])

        if sum(counts.values()) != api_total:
            raise ValueError(
                f"{topic_id}: API toplami ile "
                "yillik sayimlar eslesmiyor."
            )

        return counts

    raise RuntimeError(
        f"{topic_id} icin tekrar denemeler basarisiz."
    )


def main(limit):
    if not API_KEY:
        raise ValueError("OPENALEX_API_KEY bulunamadi.")

    topics = read_csv(TOPICS_FILE)

    if len(topics) != 500:
        raise ValueError("500 konu bekleniyor.")

    topic_ids = [row["topic_id"] for row in topics]

    if len(set(topic_ids)) != 500:
        raise ValueError("Tekrarlanan konu ID bulundu.")

    progress = read_csv(PROGRESS_FILE)

    # Tamamlanmis konulari bul
    existing = {}

    for row in progress:
        topic_id = row["topic_id"]

        if topic_id not in topic_ids:
            raise ValueError(
                f"Checkpoint icinde bilinmeyen konu: {topic_id}"
            )

        year = int(row["year"])

        if year not in YEARS:
            raise ValueError(
                f"Checkpoint icinde gecersiz yil: {year}"
            )

        key = (topic_id, year)

        if key in existing:
            raise ValueError(
                f"Tekrarlanan checkpoint kaydi: {key}"
            )

        existing[key] = row

    # Eksik yili bulunan konular bastan sorgulanir.
    completed = {
        topic_id
        for topic_id in topic_ids
        if all(
            (topic_id, year) in existing
            for year in YEARS
        )
    }

    # Yarım kalan konuların eski kayıtlarını çıkar.
    progress = [
        row for row in progress
        if row["topic_id"] in completed
    ]

    print("Toplam konu:", len(topics))
    print("Onceden tamamlanan:", len(completed))

    selected_topics = topics[:limit]

    with requests.Session() as session:
        for i, topic in enumerate(selected_topics, start=1):
            topic_id = topic["topic_id"]

            if topic_id in completed:
                print(f"[{i}/{limit}] {topic_id}: mevcut")
                continue

            counts = fetch_year_counts(session, topic_id)

            retrieved_at = datetime.now(
                timezone.utc
            ).isoformat(timespec="seconds")

            for year in sorted(YEARS):
                progress.append({
                    "topic_id": topic_id,
                    "topic_name": topic["topic_name"],
                    "domain": topic["domain"],
                    "field": topic["field"],
                    "subfield": topic["subfield"],
                    "year": year,
                    "publication_count": counts[year],
                    "retrieved_at_utc": retrieved_at,
                })

            completed.add(topic_id)
            save_csv(PROGRESS_FILE, progress)

            print(
                f"[{i}/{limit}] {topic_id}: "
                f"10 yil tamamlandi"
            )

            time.sleep(0.2)

    print("\nTamamlanan konu:", len(completed))
    print("Kaydedilen konu-yil:", len(progress))

    # Ana dosya yalnizca tum 500 konu bitince uretilir.
    if len(completed) == 500:
        order = {
            topic_id: i
            for i, topic_id in enumerate(topic_ids)
        }

        progress.sort(
            key=lambda row: (
                order[row["topic_id"]],
                int(row["year"])
            )
        )

        if len(progress) != 5000:
            raise ValueError("5000 kayit bekleniyor.")

        save_csv(OUTPUT_FILE, progress)
        print("\nANA DOSYA HAZIR:", OUTPUT_FILE)
    else:
        print("\nCheckpoint:", PROGRESS_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=3
    )

    args = parser.parse_args()

    if not 1 <= args.limit <= 500:
        parser.error("--limit 1 ile 500 arasinda olmali.")

    main(args.limit)
