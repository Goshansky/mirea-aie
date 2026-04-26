"""Сервис объяснимости предсказания."""

from __future__ import annotations

from typing import Any

import numpy as np


def _prettify_feature_name(feature_name: str) -> str:
    """Преобразует техническое имя признака в человекочитаемое."""
    normalized = feature_name.replace("num__", "").replace("cat__", "")
    return normalized.replace("_", " ")


def _format_reason(feature_name: str, contribution: float) -> str:
    """Формирует текст причины на основе вклада признака."""
    direction = "повышает риск дефолта" if contribution >= 0 else "снижает риск дефолта"
    return f"{_prettify_feature_name(feature_name)} {direction}"


def explain_with_shap(
    model: Any,
    transformed_row: np.ndarray,
    feature_names: list[str],
    top_k: int = 3,
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

    # Нормализуем форму SHAP-массива к вектору вкладов для одного объекта.
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
    return [_format_reason(name, value) for name, value in sorted_pairs[:top_k]]


def explain_with_importance(
    transformed_row: np.ndarray,
    feature_names: list[str],
    feature_importance: dict[str, float],
    top_k: int = 3,
) -> list[str]:
    """Строит fallback-объяснение по importance и направлению признака."""
    row = transformed_row[0]
    score_pairs: list[tuple[str, float]] = []
    for index, feature_name in enumerate(feature_names):
        importance = feature_importance.get(feature_name, 0.0)
        score_pairs.append((feature_name, float(row[index]) * float(importance)))

    top_pairs = sorted(score_pairs, key=lambda item: abs(item[1]), reverse=True)[:top_k]
    return [_format_reason(name, score) for name, score in top_pairs]
