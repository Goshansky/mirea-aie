"""Зависимости FastAPI."""

from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.services.inference_service import InferenceService


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    """Возвращает синглтон сервиса инференса."""
    return InferenceService(settings=get_settings())
