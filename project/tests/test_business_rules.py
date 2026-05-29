"""Тесты бизнес-правил."""

from app.core.business_rules import (
    adjust_probability_for_risk_factors,
    apply_favorable_rules,
    apply_hard_rules,
    loan_to_income_penalty,
)
from app.core.config import Settings
from app.core.decision import Decision
from app.models.schemas import PredictRequest


def test_reject_extreme_loan_to_income() -> None:
    """Отклоняет заявку с нереалистичным кредитом относительно дохода."""
    settings = Settings()
    payload = PredictRequest(
        income=1000,
        loan_amount=500_000_000_000_000,
        age=35,
        credit_history=7,
        debt_ratio=0.31,
        late_payments=0,
    )

    outcome = apply_hard_rules(payload, settings)

    assert outcome.triggered is True
    assert outcome.decision == Decision.REJECT


def test_reject_high_loan_relative_to_income() -> None:
    """5M при доходе 120k — авто-отказ (кредит >> годового дохода)."""
    settings = Settings()
    payload = PredictRequest(
        income=120_000,
        loan_amount=5_000_000,
        age=35,
        credit_history=80,
        debt_ratio=1.0,
        late_payments=0,
    )

    outcome = apply_hard_rules(payload, settings)

    assert outcome.triggered is True
    assert outcome.decision == Decision.REJECT


def test_reject_many_late_payments() -> None:
    """Много просрочек — авто-отказ."""
    settings = Settings()
    payload = PredictRequest(
        income=500_000,
        loan_amount=100_000,
        age=35,
        credit_history=5,
        debt_ratio=0.2,
        late_payments=3,
    )

    outcome = apply_hard_rules(payload, settings)

    assert outcome.triggered is True
    assert outcome.decision == Decision.REJECT


def test_approve_low_load_zero_lates() -> None:
    """Одобряет маленький кредит при высоком доходе без просрочек."""
    settings = Settings()
    payload = PredictRequest(
        income=1_200_000,
        loan_amount=10_000,
        age=35,
        credit_history=0,
        debt_ratio=0,
        late_payments=0,
    )

    outcome = apply_favorable_rules(payload, settings)

    assert outcome.triggered is True
    assert outcome.decision == Decision.APPROVE


def test_reject_extreme_age() -> None:
    """Возраст 100 — авто-отказ (превышает age_reject_max)."""
    settings = Settings()
    payload = PredictRequest(
        income=120_000,
        loan_amount=50_000,
        age=100,
        credit_history=7,
        debt_ratio=0.31,
        late_payments=0,
    )

    outcome = apply_hard_rules(payload, settings)

    assert outcome.triggered is True
    assert outcome.decision == Decision.REJECT


def test_favorable_rules_skip_old_age() -> None:
    """Возраст 100 не проходит авто-одобрение даже при низкой нагрузке."""
    settings = Settings()
    payload = PredictRequest(
        income=120_000,
        loan_amount=50_000,
        age=100,
        credit_history=7,
        debt_ratio=0.31,
        late_payments=0,
    )

    outcome = apply_favorable_rules(payload, settings)

    assert outcome.triggered is False


def test_age_increases_adjusted_pd() -> None:
    """Пожилой возраст повышает скорректированную PD."""
    settings = Settings()
    base_ml = 0.22

    young_payload = PredictRequest(
        income=120_000,
        loan_amount=50_000,
        age=35,
        credit_history=7,
        debt_ratio=0.31,
        late_payments=0,
    )
    old_payload = PredictRequest(
        income=120_000,
        loan_amount=50_000,
        age=100,
        credit_history=7,
        debt_ratio=0.31,
        late_payments=0,
    )

    young_pd, young_adj = adjust_probability_for_risk_factors(base_ml, young_payload, settings)
    old_pd, old_adj = adjust_probability_for_risk_factors(base_ml, old_payload, settings)

    assert old_pd > young_pd
    assert any("возраст" in a for a in old_adj)


def test_higher_loan_increases_penalty() -> None:
    """Больший кредит при том же доходе → больший штраф к PD."""
    settings = Settings()
    small_loan = loan_to_income_penalty(50_000 / 120_000, settings)
    big_loan = loan_to_income_penalty(5_000_000 / 120_000, settings)

    assert big_loan > small_loan


def test_adjusted_pd_higher_for_larger_loan() -> None:
    """Итоговая PD выше при большей сумме кредита."""
    settings = Settings()
    base_ml = 0.30

    small_payload = PredictRequest(
        income=120_000,
        loan_amount=50_000,
        age=35,
        credit_history=80,
        debt_ratio=1.0,
        late_payments=0,
    )
    big_payload = PredictRequest(
        income=120_000,
        loan_amount=2_000_000,
        age=35,
        credit_history=80,
        debt_ratio=1.0,
        late_payments=0,
    )

    small_pd, _ = adjust_probability_for_risk_factors(base_ml, small_payload, settings)
    big_pd, _ = adjust_probability_for_risk_factors(base_ml, big_payload, settings)

    assert big_pd > small_pd
