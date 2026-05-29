"""Тесты predict endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _assert_predict_contract(body: dict) -> None:
    """Общие проверки контракта ответа /predict."""
    assert body["decision"] in {"APPROVE", "REJECT", "REVIEW"}
    assert isinstance(body["probability"], float)
    assert 0 <= body["probability"] <= 1
    assert isinstance(body["reasons"], list)
    assert 1 <= len(body["reasons"]) <= 3
    assert all(isinstance(reason, str) and reason.strip() for reason in body["reasons"])


def test_predict_returns_scoring_response() -> None:
    """Проверяет контракт ответа /predict на ML-пути (без auto-approve)."""
    payload = {
        "income": 120000,
        "loan_amount": 200000,
        "age": 35,
        "credit_history": 7,
        "debt_ratio": 0.31,
        "late_payments": 2,
    }

    response = client.post("/predict", json=payload)
    body = response.json()

    assert response.status_code == 200
    _assert_predict_contract(body)


def test_predict_favorable_auto_approve() -> None:
    """Низкая нагрузка и ≤1 просрочка — авто-одобрение, 1–3 причины."""
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
    assert body["decision"] == "APPROVE"
    _assert_predict_contract(body)


def test_predict_hard_reject_extreme_age() -> None:
    """Возраст 100 — авто-отказ по бизнес-правилам."""
    payload = {
        "income": 120000,
        "loan_amount": 50000,
        "age": 100,
        "credit_history": 7,
        "debt_ratio": 0.31,
        "late_payments": 0,
    }

    response = client.post("/predict", json=payload)
    body = response.json()

    assert response.status_code == 200
    assert body["decision"] == "REJECT"
    _assert_predict_contract(body)
    assert any("возраст" in reason.lower() for reason in body["reasons"])


def test_predict_validation_error() -> None:
    """Проверяет валидацию входных данных."""
    invalid_payload = {
        "income": -1000,
        "loan_amount": 50000,
        "age": 17,
        "credit_history": 2,
        "debt_ratio": 0.2,
    }

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
