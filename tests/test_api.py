
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


# TEST 1: API calisiyor mu?
def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["project"] == "ResearchRadar"


# TEST 2: 500 arastirma konusu geliyor mu?
def test_topics():
    response = client.get("/topics")

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 500
    assert len(data["topics"]) == 500

    ids = [
        topic["topic_id"]
        for topic in data["topics"]
    ]

    assert len(ids) == len(set(ids))


# TEST 3: Konunun 10 yillik yayin verisi dogru mu?
def test_topic_trends():
    response = client.get("/trends/T10004")

    assert response.status_code == 200

    data = response.json()

    assert data["topic_id"] == "T10004"
    assert data["total_years"] == 10

    yearly = data["yearly_data"]

    assert len(yearly) == 10

    assert [row["year"] for row in yearly] == list(
        range(2016, 2026)
    )

    # Ilk yilin onceki yil verisi yok.
    assert yearly[0]["growth_pct"] is None

    # Son yilin buyume hesabini dogrula.
    previous_count = yearly[-2]["publication_count"]
    current_count = yearly[-1]["publication_count"]

    if previous_count > 0:
        expected_growth = round(
            (current_count - previous_count)
            / previous_count * 100,
            2
        )

        assert yearly[-1]["growth_pct"] == expected_growth


# TEST 4: Bilinmeyen konu 404 donuyor mu?
def test_unknown_topic():
    response = client.get("/trends/T99999999")

    assert response.status_code == 404


# TEST 5: Yeni Emergence Score siralamasi
def test_scores():
    response = client.get(
        "/scores",
        params={"year": 2025, "limit": 10}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["year"] == 2025
    assert data["total_scored"] == 498
    assert data["returned"] == 10

    scores = data["scores"]

    assert len(scores) == 10

    values = [
        item["research_emergence_score"]
        for item in scores
    ]

    # Skorlar buyukten kucuge sirali olmali.
    assert values == sorted(values, reverse=True)

    # Yeni 500 konuluk veri setinin ilk sirasi
    assert scores[0]["topic_name"] == (
        "Food Science and Nutritional Studies"
    )

    assert scores[0]["research_emergence_score"] == 99.52


# TEST 6: Gecersiz yil engelleniyor mu?
def test_invalid_year():
    response = client.get(
        "/scores",
        params={"year": 2018}
    )

    assert response.status_code == 400


# TEST 7: Gecersiz limit engelleniyor mu?
def test_invalid_limit():
    response = client.get(
        "/scores",
        params={"limit": 501}
    )

    assert response.status_code == 400


# TEST 8: 500 limitine izin veriliyor mu?
def test_max_limit():
    response = client.get(
        "/scores",
        params={"year": 2025, "limit": 500}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_scored"] == 498
    assert data["returned"] == 498
    assert len(data["scores"]) == 498


# TEST 9: 2026 yili henuz desteklenmiyor
def test_future_year():
    response = client.get(
        "/scores",
        params={"year": 2026}
    )

    assert response.status_code == 400
