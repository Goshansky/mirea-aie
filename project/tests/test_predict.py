"""Тесты predict endpoint."""

from fastapi.testclient import TestClient

from app.main import app


def test_predict_returns_scoring_response() -> None:
    """Проверяет базовый контракт ответа /predict."""
    client = TestClient(app)
    payload = {
        "income": 120000,
        "loan_amount": 50000,
        "age": 35,
        "credit_history": 7,
        "debt_ratio": 0.31,
        "late_payments": 1,
    }

    response = client.post("/predict", json=payload)
    body = response.json()

    assert response.status_code == 200
    assert body["decision"] in {"APPROVE", "REJECT", "REVIEW"}
    assert isinstance(body["probability"], float)
    assert 0 <= body["probability"] <= 1
    assert isinstance(body["reasons"], list)
    assert len(body["reasons"]) == 3


def test_predict_validation_error() -> None:
    """Проверяет валидацию входных данных."""
    client = TestClient(app)
    invalid_payload = {
        "income": -1000,
        "loan_amount": 50000,
        "age": 17,
        "credit_history": 2,
        "debt_ratio": 0.2,
    }

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
