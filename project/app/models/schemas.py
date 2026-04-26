"""Pydantic-схемы запросов и ответов API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Ответ health-check."""

    status: str = "ok"


class PredictRequest(BaseModel):
    """Входные признаки кредитной заявки."""

    income: float = Field(..., ge=0, description="Ежемесячный доход клиента")
    loan_amount: float = Field(..., ge=0, description="Сумма кредита")
    age: int = Field(..., ge=18, le=100, description="Возраст клиента")
    credit_history: int = Field(..., ge=0, le=80, description="Стаж кредитной истории в годах")
    debt_ratio: float = Field(..., ge=0, le=1, description="Отношение долга к доходу")
    late_payments: int = Field(default=0, ge=0, le=50, description="Количество просрочек")


class PredictResponse(BaseModel):
    """Ответ скоринга кредитной заявки."""

    decision: str
    probability: float = Field(..., ge=0, le=1)
    reasons: list[str]
