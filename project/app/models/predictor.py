"""Типы и структуры для инференса модели."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class LoadedArtifacts:
    """Контейнер загруженных артефактов модели."""

    pipeline: Pipeline
    feature_names: list[str]
    feature_importance: dict[str, float]
    artifacts_dir: Path
    raw_metadata: dict[str, Any]
