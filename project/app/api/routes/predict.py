"""Роут предсказания кредитного скоринга."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_inference_service
from app.models.schemas import PredictRequest, PredictResponse
from app.services.inference_service import InferenceService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["scoring"])


@router.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    inference_service: InferenceService = Depends(get_inference_service),
) -> PredictResponse:
    """Возвращает решение, вероятность дефолта и причины."""
    try:
        return inference_service.predict(payload)
    except ValueError as exc:
        logger.exception("Ошибка валидации/бизнес-правил: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        logger.exception("Не найдены артефакты модели: %s", exc)
        raise HTTPException(status_code=500, detail="Артефакты модели не найдены.") from exc
    except Exception as exc:
        logger.exception("Непредвиденная ошибка предсказания: %s", exc)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервиса.") from exc
