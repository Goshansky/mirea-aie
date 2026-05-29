"""Вспомогательные функции explainability для обученной модели."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Возвращает имена признаков после трансформаций."""
    try:
        names = preprocessor.get_feature_names_out().tolist()
        return [str(name) for name in names]
    except Exception:
        return []


def get_tree_feature_importance(
    model_pipeline: Pipeline,
    feature_names: list[str],
) -> dict[str, float]:
    """Возвращает importance для древовидной модели из pipeline."""
    model = model_pipeline.named_steps["model"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None or len(importances) != len(feature_names):
        return {}
    return {
        feature: float(importance)
        for feature, importance in sorted(
            zip(feature_names, importances, strict=False),
            key=lambda item: item[1],
            reverse=True,
        )
    }


def save_feature_metadata(
    feature_names: list[str],
    importances: dict[str, float],
    output_path: Path,
) -> None:
    """Сохраняет имена признаков и importances в JSON."""
    payload = {
        "feature_names": feature_names,
        "feature_importance": importances,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def calculate_shap_summary(
    model_pipeline: Pipeline,
    x_sample: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Пытается посчитать средние SHAP-вклады, иначе возвращает пустой словарь."""
    try:
        import shap  # type: ignore
    except Exception:
        return {}

    model = model_pipeline.named_steps["model"]
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_sample)
    except Exception:
        return {}

    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
    else:
        values = np.asarray(shap_values)

    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[-1] > 1 else values[:, :, 0]

    mean_abs = np.abs(values).mean(axis=0)
    mean_abs = np.asarray(mean_abs).reshape(-1)
    if len(mean_abs) != len(feature_names):
        return {}

    return {
        feature: float(value)
        for feature, value in sorted(
            zip(feature_names, mean_abs, strict=False),
            key=lambda item: item[1],
            reverse=True,
        )
    }
