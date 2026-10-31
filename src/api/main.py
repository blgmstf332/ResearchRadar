
import csv
import math
from pathlib import Path

from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="ResearchRadar API",
    description="Akademik arastirma trendleri analiz sistemi",
    version="1.0.0"
)

ROOT = Path(__file__).resolve().parents[2]

TOPICS_FILE = ROOT / "data/samples/topic_candidates_120.csv"
COUNTS_FILE = ROOT / "data/samples/topic_year_counts_120.csv"


# 1. API saglik kontrolu
@app.get("/health")
def health():
    return {
        "status": "ok",
        "project": "ResearchRadar",
        "message": "API basariyla calisiyor"
    }


# 2. Tum arastirma konularini listele
@app.get("/topics")
def get_topics():

    if not TOPICS_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail="Arastirma konulari dosyasi bulunamadi."
        )

    with open(
        TOPICS_FILE,
        mode="r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        topics = list(csv.DictReader(file))

    return {
        "total": len(topics),
        "topics": topics
    }


# 3. Secilen konunun yillik yayin trendini getir
@app.get("/trends/{topic_id}")
def get_topic_trends(topic_id: str):

    if not COUNTS_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail="Yillik yayin verileri bulunamadi."
        )

    topic_id = topic_id.strip().upper()

    with open(
        COUNTS_FILE,
        mode="r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        topic_rows = [
            row for row in reader
            if row["topic_id"].strip().upper() == topic_id
        ]

    if not topic_rows:
        raise HTTPException(
            status_code=404,
            detail=f"{topic_id} konusuna ait veri bulunamadi."
        )

    # Verileri yila gore sirala
    topic_rows.sort(key=lambda row: int(row["year"]))

    yearly_data = []
    previous_count = None

    for row in topic_rows:

        year = int(row["year"])
        count = int(row["publication_count"])

        # Ilk yil icin onceki yil olmadigindan
        # buyume orani hesaplanamaz.
        growth_pct = None

        if previous_count is not None and previous_count > 0:
            growth_pct = round(
                (count - previous_count)
                / previous_count * 100,
                2
            )

        yearly_data.append({
            "year": year,
            "publication_count": count,
            "growth_pct": growth_pct
        })

        previous_count = count

    return {
        "topic_id": topic_id,
        "total_years": len(yearly_data),
        "yearly_data": yearly_data
    }

# 4. Research Emergence Score siralamasi
@app.get("/scores")
def get_scores(year: int = 2025, limit: int = 10):

    scores_file = (
        ROOT / "data/samples/topic_year_scores_v1.csv"
    )

    if not scores_file.exists():
        raise HTTPException(
            status_code=503,
            detail="Emergence Score veri dosyasi bulunamadi."
        )

    if year < 2018 or year > 2025:
        raise HTTPException(
            status_code=400,
            detail="Yil 2018-2025 arasinda olmali."
        )

    if limit < 1 or limit > 120:
        raise HTTPException(
            status_code=400,
            detail="Limit 1-120 arasinda olmali."
        )

    scores = []

    with open(
        scores_file,
        mode="r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if int(row["year"]) != year:
                continue

            raw_score = row["research_emergence_score"]

            # Skoru bulunmayan konulari atla
            if not raw_score:
                continue

            score = float(raw_score)

            if not math.isfinite(score):
                continue

            scores.append({
                "topic_id": row["topic_id"],
                "topic_name": row["topic_name"],
                "domain": row["domain"],
                "year": int(row["year"]),
                "growth_1y_pct": float(row["growth_1y_pct"]),
                "citation_velocity_1y": float(
                    row["citation_velocity_1y"]
                ),
                "research_emergence_score": score
            })

    # En yuksek puandan en dusuge sirala
    scores.sort(
        key=lambda x: x["research_emergence_score"],
        reverse=True
    )

    return {
        "year": year,
        "total_scored": len(scores),
        "returned": min(limit, len(scores)),
        "scores": scores[:limit]
    }
