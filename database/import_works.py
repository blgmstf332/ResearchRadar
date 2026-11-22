
import csv
import getpass
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb


CSV_PATH = Path(
    "data/processed/work_details_500_topics_n30_cleaned.csv"
)

BATCH_SIZE = 500

COLUMNS = [
    "work_id",
    "topic_id",
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
    "missing_title",
    "missing_doi",
    "missing_abstract",
]

BOOLEAN_COLUMNS = {
    "has_abstract",
    "is_retracted",
    "missing_title",
    "missing_doi",
    "missing_abstract",
}

INTEGER_COLUMNS = {
    "publication_year",
    "cited_by_count",
}


def convert_value(column, value):
    """CSV degerini PostgreSQL icin uygun tipe cevir."""

    if value is None or value.strip() == "":
        return None

    value = value.strip()

    if column in INTEGER_COLUMNS:
        number = Decimal(value)

        if not number.is_finite():
            raise ValueError(f"Gecersiz sayi: {value}")

        if number != number.to_integral_value():
            raise ValueError(f"Tam sayi bekleniyordu: {value}")

        return int(number)

    if column in BOOLEAN_COLUMNS:
        if value.lower() == "true":
            return True

        if value.lower() == "false":
            return False

        raise ValueError(f"Gecersiz boolean: {value}")

    if column == "publication_date":
        return date.fromisoformat(value)

    if column == "counts_by_year_json":
        return Jsonb(json.loads(value))

    return value


def create_upsert_query():
    """Ayni work_id tekrar gelirse kaydi guncelle."""

    column_sql = sql.SQL(", ").join(
        sql.Identifier(column)
        for column in COLUMNS
    )

    placeholders = sql.SQL(", ").join(
        sql.Placeholder()
        for _ in COLUMNS
    )

    updates = sql.SQL(", ").join(
        sql.SQL("{} = EXCLUDED.{}").format(
            sql.Identifier(column),
            sql.Identifier(column),
        )
        for column in COLUMNS
        if column != "work_id"
    )

    return sql.SQL(
        "INSERT INTO works ({}) VALUES ({}) "
        "ON CONFLICT (work_id) DO UPDATE SET {}"
    ).format(
        column_sql,
        placeholders,
        updates,
    )


def main():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(
            f"CSV dosyasi bulunamadi: {CSV_PATH}"
        )

    print(f"Okunacak dosya: {CSV_PATH}")

    password = getpass.getpass("PostgreSQL sifresi: ")

    processed = 0
    seen_ids = set()
    batch = []

    # Tum aktarim tek transaction icinde gerceklesir.
    # Hata olursa degisiklikler geri alinir.
    with psycopg.connect(
        host="localhost",
        port=5432,
        dbname="researchradar",
        user="postgres",
        password=password,
    ) as connection:

        with connection.cursor() as cursor:

            # Veritabanindaki gecerli konu kimlikleri.
            cursor.execute("SELECT topic_id FROM topics")

            valid_topic_ids = {
                row[0]
                for row in cursor.fetchall()
            }

            print(
                f"Veritabanindaki konu sayisi: "
                f"{len(valid_topic_ids)}"
            )

            query = create_upsert_query()

            with CSV_PATH.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as file:

                reader = csv.DictReader(file)

                missing_columns = (
                    set(COLUMNS)
                    - set(reader.fieldnames or [])
                )

                if missing_columns:
                    raise ValueError(
                        f"Eksik CSV sutunlari: "
                        f"{missing_columns}"
                    )

                for line_number, record in enumerate(
                    reader,
                    start=2,
                ):
                    try:
                        work_id = record["work_id"]
                        topic_id = record["topic_id"]

                        if not work_id or not work_id.strip():
                            raise ValueError(
                                "Bos work_id"
                            )

                        if not topic_id or not topic_id.strip():
                            raise ValueError(
                                "Bos topic_id"
                            )

                        work_id = work_id.strip()
                        topic_id = topic_id.strip()

                        if work_id in seen_ids:
                            raise ValueError(
                                f"Tekrarli work_id: {work_id}"
                            )

                        if topic_id not in valid_topic_ids:
                            raise ValueError(
                                f"Konusu bulunamadi: {topic_id}"
                            )

                        seen_ids.add(work_id)

                        record["work_id"] = work_id
                        record["topic_id"] = topic_id

                        values = tuple(
                            convert_value(
                                column,
                                record[column],
                            )
                            for column in COLUMNS
                        )

                        batch.append(values)

                    except (ValueError, KeyError, TypeError) as error:
                        raise ValueError(
                            f"CSV satiri {line_number}: {error}"
                        ) from error

                    if len(batch) >= BATCH_SIZE:
                        cursor.executemany(query, batch)
                        processed += len(batch)
                        batch.clear()

                        if processed % 10000 == 0:
                            print(
                                f"Islenen makale: {processed}"
                            )

                if batch:
                    cursor.executemany(query, batch)
                    processed += len(batch)
                    batch.clear()

            if processed == 0:
                raise ValueError("CSV dosyasinda makale yok.")

            print(f"Toplam islenen makale: {processed}")

        # Burada transaction commit edilir.

    print("Works aktarimi basariyla tamamlandi!")


if __name__ == "__main__":
    main()
