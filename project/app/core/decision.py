"""Бизнес-правила принятия решения по заявке."""

from __future__ import annotations

from enum import Enum


class Decision(str, Enum):
    """Возможные решения по заявке."""

    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REVIEW = "REVIEW"


def resolve_decision(pd_probability: float, threshold_approve: float, threshold_reject: float) -> Decision:
    """Возвращает решение на основе порогов PD."""
    if threshold_approve >= threshold_reject:
        raise ValueError("Порог approve должен быть строго меньше порога reject.")

    if pd_probability < threshold_approve:
        return Decision.APPROVE
    if pd_probability > threshold_reject:
        return Decision.REJECT
    return Decision.REVIEW
