
import csv
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("OPENALEX_API_KEY")

INPUT_FILE = ROOT / "data/samples/topic_candidates_120.csv"
OUTPUT_FILE = ROOT / "data/samples/topic_candidates_500.csv"

API_URL = "https://api.openalex.org/topics"

DOMAINS = [
    "Health Sciences",
    "Life Sciences",
    "Physical Sciences",
    "Social Sciences",
]

TARGET_PER_DOMAIN = 125
MIN_WORKS = 100
SEED = 42


def read_existing_topics():
    with open(INPUT_FILE, encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    ids = [row["topic_id"] for row in rows]

    if len(rows) != 120 or len(set(ids)) != 120:
        raise ValueError("Eski 120 konuda kayit/ID sorunu var.")

    counts = Counter(row["domain"] for row in rows)

    for domain in DOMAINS:
        if counts[domain] != 30:
            raise ValueError(f"Beklenmeyen domain dagilimi: {domain}")

    return rows


def download_topic_catalog():
    if not API_KEY:
        raise ValueError("OPENALEX_API_KEY .env dosyasinda bulunamadi.")

    topics = []
    cursor = "*"

    with requests.Session() as session:
        for page in range(1, 101):
            response = None

            for attempt in range(5):
                try:
                    response = session.get(
                        API_URL,
                        params={
                            "api_key": API_KEY,
                            "per_page": 100,
                            "cursor": cursor,
                        },
                        timeout=40,
                    )

                    if response.status_code in (429, 500, 502, 503, 504):
                        time.sleep(min(2 ** attempt, 16))
                        continue

                    response.raise_for_status()
                    break

                except requests.RequestException:
                    if attempt == 4:
                        raise
                    time.sleep(min(2 ** attempt, 16))

            if response is None or not response.ok:
                raise RuntimeError("OpenAlex istegi basarisiz.")

            payload = response.json()
            results = payload.get("results", [])

            if not results:
                break

            for item in results:
                domain = item.get("domain") or {}
                field = item.get("field") or {}
                subfield = item.get("subfield") or {}

                if domain.get("display_name") not in DOMAINS:
                    continue

                topics.append({
                    "topic_id": item["id"].rstrip("/").split("/")[-1],
                    "topic_name": item.get("display_name", ""),
                    "domain": domain["display_name"],
                    "field": field.get("display_name", ""),
                    "subfield": subfield.get("display_name", ""),
                    "works_count": int(item.get("works_count") or 0),
                })

            print(f"Sayfa {page}: {len(topics)} konu okundu")

            cursor = payload.get("meta", {}).get("next_cursor")

            if not cursor:
                break

            time.sleep(0.15)
        else:
            raise RuntimeError("100 sayfa sinirina ulasildi.")

    if not topics:
        raise RuntimeError("OpenAlex konulari alinamadi.")

    ids = [row["topic_id"] for row in topics]
    if len(ids) != len(set(ids)):
        raise ValueError("OpenAlex katalogunda tekrar eden ID.")

    print("OpenAlex katalog toplam:", len(topics))
    return topics


def select_500(existing, catalog):
    selected = list(existing)
    used_ids = {row["topic_id"] for row in selected}

    rng = random.Random(SEED)

    candidates = defaultdict(list)

    for topic in catalog:
        if topic["topic_id"] in used_ids:
            continue
        if topic["works_count"] < MIN_WORKS:
            continue
        if not topic["field"]:
            continue

        candidates[(topic["domain"], topic["field"])].append(topic)

    # Secimi sabit bir seed ile tekrarlanabilir yap.
    for key in sorted(candidates):
        candidates[key].sort(key=lambda x: x["topic_id"])
        rng.shuffle(candidates[key])

    for domain in DOMAINS:
        domain_rows = [
            row for row in selected
            if row["domain"] == domain
        ]

        field_counts = Counter(
            row["field"] for row in domain_rows
        )

        while len(domain_rows) < TARGET_PER_DOMAIN:
            available_fields = [
                field
                for d, field in candidates
                if d == domain and candidates[(d, field)]
            ]

            if not available_fields:
                raise RuntimeError(
                    f"{domain} icin yeterli yeni konu bulunamadi."
                )

            # Daha az temsil edilen field'i sec.
            chosen_field = min(
                available_fields,
                key=lambda f: (field_counts[f], f)
            )

            topic = candidates[(domain, chosen_field)].pop()

            selected.append(topic)
            domain_rows.append(topic)
            used_ids.add(topic["topic_id"])
            field_counts[chosen_field] += 1

    if len(selected) != 500 or len(used_ids) != 500:
        raise RuntimeError("500 benzersiz konu olusturulamadi.")

    return selected


def main():
    existing = read_existing_topics()

    print("Mevcut konu:", len(existing))
    print("OpenAlex konu katalogu aliniyor...")

    catalog = download_topic_catalog()
    selected = select_500(existing, catalog)

    columns = [
        "topic_id",
        "topic_name",
        "domain",
        "field",
        "subfield",
        "works_count"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(selected)

    print("\n--- RESEARCHRADAR 500 TOPICS ---")
    print("Toplam konu:", len(selected))
    print("Benzersiz ID:", len(set(
        row["topic_id"] for row in selected
    )))
    print("Korunan eski konu:", len(existing))
    print("Eklenen yeni konu:", len(selected) - len(existing))

    print("\nDomain dagilimi:")
    for domain, count in sorted(
        Counter(row["domain"] for row in selected).items()
    ):
        print(f"{domain}: {count}")

    print("\nField dagilimi:")
    for (domain, field), count in sorted(
        Counter(
            (row["domain"], row["field"])
            for row in selected
        ).items()
    ):
        print(f"{domain} / {field}: {count}")

    print("\nKaydedildi:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
