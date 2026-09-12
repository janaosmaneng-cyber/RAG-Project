from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }


def test_query_happy_path():
    response = client.post(
        "/query",
        data={
            "question": "What are the main symptoms of insomnia?"
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert "answer" in body
    assert "sources" in body
    assert isinstance(body["sources"], list)


def test_query_invalid_input():
    response = client.post(
        "/query",
        data={
            "question": ""
        },
    )

    assert response.status_code == 422