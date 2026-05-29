"""Сервис объяснимости предсказания."""

from __future__ import annotations

from typing import Any

import numpy as np

# Человекочитаемые названия признаков для explainability (RU).
FEATURE_LABELS_RU: dict[str, str] = {
    "income": "доход",
    "loan_amount": "сумма кредита",
    "age": "возраст",
    "credit_history": "число кредитных линий",
    "debt_ratio": "долговая нагрузка",
    "late_payments": "просрочки",
    "loan_to_income": "кредит к доходу",
}


def _normalize_feature_name(feature_name: str) -> str:
    """Убирает префиксы sklearn ColumnTransformer."""
    return feature_name.replace("num__", "").replace("cat__", "")


def _format_reason(feature_name: str, contribution: float, raw_value: float | None = None) -> str:
    """Формирует текст причины на основе вклада признака."""
    normalized = _normalize_feature_name(feature_name)
    label = FEATURE_LABELS_RU.get(normalized, normalized.replace("_", " "))

    if raw_value is not None:
        if normalized == "late_payments" and raw_value == 0:
            label = "отсутствие просрочек"
        elif normalized == "debt_ratio" and raw_value == 0:
            label = "нулевая долговая нагрузка"
        elif normalized == "credit_history" and raw_value == 0:
            label = "отсутствие кредитных линий"

    if contribution > 0:
        return f"{label} повышает риск дефолта"
    if contribution < 0:
        return f"{label} снижает риск дефолта"
    return f"{label} — нейтральный фактор"


def explain_with_shap(
    model: Any,
    transformed_row: np.ndarray,
    feature_names: list[str],
    top_k: int = 3,
    raw_values: dict[str, float] | None = None,
) -> list[str]:
    """Пытается построить объяснение через SHAP для одной заявки."""
    try:
        import shap  # type: ignore
    except Exception:
        return []

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed_row)
    except Exception:
        return []

    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
    else:
        values = np.asarray(shap_values)

    if values.ndim == 3:
        row_values = values[0, :, 1] if values.shape[-1] > 1 else values[0, :, 0]
    elif values.ndim == 2:
        row_values = values[0]
    elif values.ndim == 1:
        row_values = values
    else:
        return []

    row_values = np.asarray(row_values, dtype=float)
    if len(row_values) != len(feature_names):
        return []

    sorted_pairs = sorted(
        zip(feature_names, row_values, strict=False),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    return [
        _format_reason(
            name,
            value,
            raw_values.get(_normalize_feature_name(name)) if raw_values else None,
        )
        for name, value in sorted_pairs[:top_k]
    ]


def explain_with_importance(
    transformed_row: np.ndarray,
    feature_names: list[str],
    feature_importance: dict[str, float],
    top_k: int = 3,
    raw_values: dict[str, float] | None = None,
) -> list[str]:
    """Строит fallback-объяснение по importance и направлению признака."""
    row = transformed_row[0]
    score_pairs: list[tuple[str, float]] = []
    for index, feature_name in enumerate(feature_names):
        importance = feature_importance.get(feature_name, 0.0)
        score_pairs.append((feature_name, float(row[index]) * float(importance)))

    top_pairs = sorted(score_pairs, key=lambda item: abs(item[1]), reverse=True)[:top_k]
    return [
        _format_reason(
            name,
            score,
            raw_values.get(_normalize_feature_name(name)) if raw_values else None,
        )
        for name, score in top_pairs
    ]
