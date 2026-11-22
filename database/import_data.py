
import argparse
import csv
import getpass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import psycopg
from psycopg import sql


# Sayisal sutunlar
INTEGER_COLUMNS = {
    "year",
    "works_count",
    "publication_count",
    "prev_year_count",
    "count_2y_ago",
    "count_3y_ago",
    "citation_sample_size",
}

FLOAT_COLUMNS = {
    "growth_1y_pct",
    "growth_2y_pct",
    "growth_3y_pct",
    "growth_acceleration",
    "growth_stability",
    "citation_velocity_1y",
    "citation_velocity_median",
    "early_citation_mean",
    "growth_score",
    "citation_score",
    "research_emergence_score",
}


def convert_value(column, value):
    """CSV degerlerini PostgreSQL'e uygun Python tiplerine cevirir."""

    if value is None or value.strip() == "":
        return None

    value = value.strip()

    if column in INTEGER_COLUMNS:
        number = Decimal(value)

        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError(
                f"{column} alaninda gecersiz tam sayi: {value}"
            )

        return int(number)

    if column in FLOAT_COLUMNS:
        number = float(value)

        if not Decimal(str(number)).is_finite():
            raise ValueError(
                f"{column} alaninda gecersiz sayi: {value}"
            )

        return number

    if column == "is_emerging_candidate":
        if value.lower() == "true":
            return True

        if value.lower() == "false":
            return False

        raise ValueError(
            f"Gecersiz boolean degeri: {value}"
        )

    if column == "retrieved_at_utc":
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

    return value


def read_csv_rows(file_path, columns, key_columns):
    """CSV dosyasini okur ve benzersiz anahtarlari kontrol eder."""

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dosya bulunamadi: {file_path}"
        )

    rows = []
    seen_keys = set()

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        missing_columns = set(columns) - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(
                f"{file_path.name} eksik sutunlar: {missing_columns}"
            )

        for line_number, record in enumerate(reader, start=2):

            converted = {
                column: convert_value(column, record[column])
                for column in columns
            }

            key = tuple(
                converted[column]
                for column in key_columns
            )

            if None in key:
                raise ValueError(
                    f"{file_path.name}, satir {line_number}: bos anahtar"
                )

            if key in seen_keys:
                raise ValueError(
                    f"{file_path.name}, satir {line_number}: "
                    f"tekrarli anahtar {key}"
                )

            seen_keys.add(key)

            rows.append(
                tuple(
                    converted[column]
                    for column in columns
                )
            )

    return rows


def create_upsert_query(table, columns, key_columns):
    """Ayni kayit tekrar gelirse guncelleyen SQL sorgusu olusturur."""

    column_names = sql.SQL(", ").join(
        sql.Identifier(column)
        for column in columns
    )

    placeholders = sql.SQL(", ").join(
        sql.Placeholder()
        for _ in columns
    )

    conflict_columns = sql.SQL(", ").join(
        sql.Identifier(column)
        for column in key_columns
    )

    updates = sql.SQL(", ").join(
        sql.SQL("{} = EXCLUDED.{}").format(
            sql.Identifier(column),
            sql.Identifier(column),
        )
        for column in columns
        if column not in key_columns
    )

    return sql.SQL(
        "INSERT INTO {} ({}) VALUES ({}) "
        "ON CONFLICT ({}) DO UPDATE SET {}"
    ).format(
        sql.Identifier(table),
        column_names,
        placeholders,
        conflict_columns,
        updates,
    )


def main():
    parser = argparse.ArgumentParser(
        description="ResearchRadar PostgreSQL CSV aktarimi"
    )

    parser.add_argument(
        "--topics",
        type=Path,
        default=Path(
            "data/samples/topic_candidates_500.csv"
        ),
    )

    parser.add_argument(
        "--counts",
        type=Path,
        default=Path(
            "data/samples/topic_year_counts_500.csv"
        ),
    )

    parser.add_argument(
        "--scores",
        type=Path,
        default=Path(
            "data/samples/topic_year_scores_v2_500.csv"
        ),
    )

    args = parser.parse_args()

    configurations = [
        (
            "topics",
            args.topics,
            [
                "topic_id",
                "topic_name",
                "domain",
                "field",
                "subfield",
                "works_count",
            ],
            ["topic_id"],
        ),
        (
            "topic_year_counts",
            args.counts,
            [
                "topic_id",
                "year",
                "publication_count",
                "retrieved_at_utc",
            ],
            ["topic_id", "year"],
        ),
        (
            "topic_year_scores",
            args.scores,
            [
                "topic_id",
                "year",
                "prev_year_count",
                "count_2y_ago",
                "count_3y_ago",
                "growth_1y_pct",
                "growth_2y_pct",
                "growth_3y_pct",
                "growth_acceleration",
                "growth_stability",
                "citation_sample_size",
                "citation_velocity_1y",
                "citation_velocity_median",
                "early_citation_mean",
                "growth_score",
                "citation_score",
                "research_emergence_score",
                "is_emerging_candidate",
            ],
            ["topic_id", "year"],
        ),
    ]

    # Once tum CSV dosyalarini kontrol et.
    prepared_data = []

    for table, file_path, columns, key_columns in configurations:

        rows = read_csv_rows(
            file_path,
            columns,
            key_columns
        )

        prepared_data.append(
            (table, columns, key_columns, rows)
        )

        print(
            f"CSV kontrol edildi: {table} -> {len(rows)} kayit"
        )

    # Sifre kodda saklanmaz.
    password = getpass.getpass(
        "PostgreSQL sifresi: "
    )

    # Tum aktarimlar tek transaction icinde yapilir.
    with psycopg.connect(
        host="localhost",
        port=5432,
        dbname="researchradar",
        user="postgres",
        password=password,
    ) as connection:

        with connection.cursor() as cursor:

            for table, columns, key_columns, rows in prepared_data:

                query = create_upsert_query(
                    table,
                    columns,
                    key_columns
                )

                cursor.executemany(
                    query,
                    rows
                )

                print(
                    f"Aktarildi: {table} -> {len(rows)} kayit"
                )

    print(
        "PostgreSQL veri aktarimi basariyla tamamlandi!"
    )


if __name__ == "__main__":
    main()
