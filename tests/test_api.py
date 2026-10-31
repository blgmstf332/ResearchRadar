
from fastapi.testclient import TestClient

from src.api.main import app

# FastAPI uygulamasini test etmek icin
# sanal bir HTTP istemcisi olusturuyoruz.
client = TestClient(app)


# TEST 1: API calisiyor mu?
def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["project"] == "ResearchRadar"


# TEST 2: 120 arastirma konusu geliyor mu?
def test_topics():
    response = client.get("/topics")

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 120
    assert len(data["topics"]) == 120


# TEST 3: Bilinen konunun yillik verileri dogru mu?
def test_topic_trends():
    response = client.get("/trends/T10004")

    assert response.status_code == 200

    data = response.json()

    assert data["topic_id"] == "T10004"
    assert data["total_years"] == 8

    yearly = data["yearly_data"]

    assert yearly[0]["year"] == 2018
    assert yearly[-1]["year"] == 2025

    assert yearly[-1]["publication_count"] == 9106
    assert yearly[-1]["growth_pct"] == 19.38


# TEST 4: Bilinmeyen konu icin 404 donuyor mu?
def test_unknown_topic():
    response = client.get("/trends/T99999999")

    assert response.status_code == 404


# TEST 5: Emergence Score siralamasi dogru mu?
def test_scores():
    response = client.get(
        "/scores",
        params={"year": 2025, "limit": 10}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["year"] == 2025
    assert data["total_scored"] == 119
    assert data["returned"] == 10

    scores = data["scores"]

    assert len(scores) == 10

    values = [
        item["research_emergence_score"]
        for item in scores
    ]

    # Skorlar buyukten kucuge sirali olmali
    assert values == sorted(values, reverse=True)

    # Ilk konunun dogrulugu
    assert scores[0]["topic_id"] == "T12492"
    assert scores[0]["research_emergence_score"] == 92.67


# TEST 6: Gecersiz yil engelleniyor mu?
def test_invalid_year():
    response = client.get(
        "/scores",
        params={"year": 2017}
    )

    assert response.status_code == 400


# TEST 7: Gecersiz limit engelleniyor mu?
def test_invalid_limit():
    response = client.get(
        "/scores",
        params={"limit": 200}
    )

    assert response.status_code == 400
