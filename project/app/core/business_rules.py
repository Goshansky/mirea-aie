"""Бизнес-правила до ML-скоринга: авто-одобрение и авто-отказ."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings
from app.core.decision import Decision
from app.models.schemas import PredictRequest


@dataclass(frozen=True)
class BusinessRuleOutcome:
    """Результат срабатывания бизнес-правила."""

    triggered: bool
    decision: Decision | None = None
    probability: float | None = None
    reasons: tuple[str, ...] = ()


def _loan_to_income(payload: PredictRequest) -> float:
    """Считает отношение суммы кредита к месячному доходу."""
    return payload.loan_amount / max(payload.income, 1)


def late_payment_penalty(late_payments: int, settings: Settings) -> float:
    """
    Добавка к PD за просрочки — монотонно растёт, но без «обрыва» на 0→1.

    1 просрочка не должна давать скачок с 3% до 36%, если профиль сильный.
    """
    if late_payments <= 0:
        return 0.0
    penalty = late_payments * settings.late_payment_penalty
    return min(settings.late_payment_penalty_cap, penalty)


def loan_to_income_penalty(loan_to_income: float, settings: Settings) -> float:
    """
    Штраф к PD за высокий кредит относительно дохода.

    Чем больше сумма кредита (при том же доходе), тем выше итоговый риск.
    """
    if loan_to_income <= settings.loan_penalty_free_threshold:
        return 0.0
    excess = loan_to_income - settings.loan_penalty_free_threshold
    penalty = excess * settings.loan_penalty_rate
    return min(settings.loan_penalty_cap, penalty)


def age_penalty(age: int, settings: Settings) -> float:
    """
    Штраф к PD за возраст вне «рабочего» диапазона.

    Слишком молодой — меньше стабильности, слишком старший — выше риск дефолта.
    """
    penalty = 0.0
    if age < settings.age_penalty_young_threshold:
        penalty += (settings.age_penalty_young_threshold - age) * settings.age_penalty_rate
    if age > settings.age_penalty_old_threshold:
        penalty += (age - settings.age_penalty_old_threshold) * settings.age_penalty_rate
    return min(settings.age_penalty_cap, penalty)


def debt_ratio_penalty(debt_ratio: float, settings: Settings) -> float:
    """Штраф за высокую долговую нагрузку."""
    if debt_ratio <= settings.debt_penalty_start:
        return 0.0
    excess = debt_ratio - settings.debt_penalty_start
    return min(settings.debt_penalty_cap, excess * settings.debt_penalty_rate)


def credit_history_penalty(credit_history: int, settings: Settings) -> float:
    """
    Штраф к PD за отсутствие или избыток открытых кредитных линий.

    0 линий — thin file (нет кредитной истории).
    Много линий — перегруз параллельными обязательствами.
    """
    penalty = 0.0
    if credit_history <= 0:
        penalty += settings.credit_history_thin_penalty
    elif credit_history > settings.credit_history_overload_start:
        excess = credit_history - settings.credit_history_overload_start
        penalty += excess * settings.credit_history_penalty_rate
    return min(settings.credit_history_penalty_cap, penalty)


def adjust_probability_for_risk_factors(
    ml_probability: float,
    payload: PredictRequest,
    settings: Settings,
) -> tuple[float, list[str]]:
    """Корректирует ML-вероятность с учётом просрочек, кредита, долга, возраста и кредитных линий."""
    loan_to_income = _loan_to_income(payload)
    adjustments: list[str] = []

    age_pen = age_penalty(payload.age, settings)
    if age_pen > 0:
        adjustments.append(f"возраст ({payload.age} лет): +{age_pen * 100:.0f} п.п. к риску")

    credit_pen = credit_history_penalty(payload.credit_history, settings)
    if credit_pen > 0:
        if payload.credit_history <= 0:
            adjustments.append(
                f"нет кредитной истории (0 линий): +{credit_pen * 100:.0f} п.п. к риску"
            )
        else:
            adjustments.append(
                f"много открытых линий ({payload.credit_history}): +{credit_pen * 100:.0f} п.п. к риску"
            )

    late_pen = late_payment_penalty(payload.late_payments, settings)
    if late_pen > 0:
        adjustments.append(
            f"просрочки ({payload.late_payments}): +{late_pen * 100:.0f} п.п. к риску"
        )

    loan_pen = loan_to_income_penalty(loan_to_income, settings)
    if loan_pen > 0:
        adjustments.append(
            f"кредит {loan_to_income:.1f}× месячного дохода: +{loan_pen * 100:.0f} п.п. к риску"
        )

    debt_pen = debt_ratio_penalty(payload.debt_ratio, settings)
    if debt_pen > 0:
        adjustments.append(
            f"высокая долговая нагрузка ({payload.debt_ratio:.2f}): +{debt_pen * 100:.0f} п.п. к риску"
        )

    adjusted = min(0.99, ml_probability + age_pen + credit_pen + late_pen + loan_pen + debt_pen)
    return adjusted, adjustments


def adjust_probability_for_late_payments(
    ml_probability: float,
    late_payments: int,
    settings: Settings,
) -> float:
    """Корректирует ML-вероятность с учётом просрочек (совместимость)."""
    return min(0.99, ml_probability + late_payment_penalty(late_payments, settings))


def apply_favorable_rules(payload: PredictRequest, settings: Settings) -> BusinessRuleOutcome:
    """
    Авто-одобрение для явно безопасных заявок.

    Допускает 1 просрочку при очень низкой кредитной нагрузке (реалистичнее, чем 0/1 обрыв).
    """
    if payload.income < settings.min_income:
        return BusinessRuleOutcome(triggered=False)

    loan_to_income = _loan_to_income(payload)
    is_low_load = loan_to_income <= settings.auto_approve_max_loan_to_income
    is_acceptable_debt = payload.debt_ratio <= settings.auto_approve_max_debt_ratio
    is_acceptable_lates = payload.late_payments <= settings.auto_approve_max_late_payments
    is_acceptable_age = settings.auto_approve_min_age <= payload.age <= settings.auto_approve_max_age
    is_acceptable_credit_history = (
        settings.auto_approve_min_credit_history
        <= payload.credit_history
        <= settings.auto_approve_max_credit_history
    )

    if not (
        is_low_load
        and is_acceptable_debt
        and is_acceptable_lates
        and is_acceptable_age
        and is_acceptable_credit_history
    ):
        return BusinessRuleOutcome(triggered=False)

    base_pd = max(0.03, loan_to_income * 0.5 + payload.debt_ratio * 0.1)
    estimated_pd = min(
        0.22,
        base_pd
        + late_payment_penalty(payload.late_payments, settings)
        + age_penalty(payload.age, settings)
        + credit_history_penalty(payload.credit_history, settings),
    )

    reasons: list[str] = [
        f"кредит составляет {loan_to_income:.2f} от месячного дохода — низкая нагрузка",
    ]
    if payload.late_payments == 0:
        reasons.append("нет просрочек")
    else:
        reasons.append(
            f"единичная просрочка ({payload.late_payments}) при низкой нагрузке — умеренный риск"
        )
    if payload.income >= 500_000:
        reasons.append("высокий подтверждённый доход")

    return BusinessRuleOutcome(
        triggered=True,
        decision=Decision.APPROVE,
        probability=estimated_pd,
        reasons=tuple(reasons[:3]),
    )


def apply_hard_rules(payload: PredictRequest, settings: Settings) -> BusinessRuleOutcome:
    """
    Авто-отказ для явно нереалистичных или рискованных комбинаций.

    Много просрочек — сильный сигнал дефолта в реальной практике.
    """
    reasons: list[str] = []
    loan_to_income = _loan_to_income(payload)

    if payload.late_payments >= settings.late_payments_reject:
        reasons.append(
            f"много просрочек ({payload.late_payments}): "
            f"порог авто-отказа — {settings.late_payments_reject}"
        )

    if payload.age > settings.age_reject_max:
        reasons.append(
            f"возраст ({payload.age}) превышает допустимый предел ({settings.age_reject_max} лет)"
        )

    if payload.age < settings.age_reject_min:
        reasons.append(
            f"возраст ({payload.age}) ниже допустимого предела ({settings.age_reject_min} лет)"
        )

    if payload.credit_history >= settings.credit_history_reject_max:
        reasons.append(
            f"слишком много открытых кредитных линий ({payload.credit_history}): "
            f"лимит — {settings.credit_history_reject_max - 1}"
        )

    if payload.income < settings.min_income:
        reasons.append(
            f"доход ({payload.income:,.0f} ₽) ниже минимального порога ({settings.min_income:,.0f} ₽)"
        )

    if loan_to_income > settings.max_loan_to_income:
        reasons.append(
            f"кредит превышает допустимую нагрузку: {loan_to_income:,.0f} месячных доходов "
            f"(лимит {settings.max_loan_to_income:.0f})"
        )

    implied_load = min(loan_to_income / settings.max_loan_to_income, 1.0)
    if loan_to_income > 12 and payload.debt_ratio + 0.15 < implied_load:
        reasons.append(
            "несогласованность: низкая долговая нагрузка при очень большой сумме кредита относительно дохода"
        )

    if not reasons:
        return BusinessRuleOutcome(triggered=False)

    reject_probability = min(
        0.99,
        0.80
        + age_penalty(payload.age, settings)
        + credit_history_penalty(payload.credit_history, settings)
        + late_payment_penalty(payload.late_payments, settings)
        + 0.03 * len(reasons),
    )

    return BusinessRuleOutcome(
        triggered=True,
        decision=Decision.REJECT,
        probability=reject_probability,
        reasons=tuple(reasons),
    )
