"""Точка входа FastAPI приложения."""

from __future__ import annotations

import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request

from app.api.routes.health import router as health_router
from app.api.routes.predict import router as predict_router
from app.core.config import get_settings
from app.core.logging import configure_logging

settings = get_settings()
configure_logging(settings.app_log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)
app.include_router(health_router)
app.include_router(predict_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Логирует базовые метрики по каждому HTTP-запросу."""
    request_id = str(uuid4())
    start_time = time.perf_counter()
    logger.info("request_started id=%s method=%s path=%s", request_id, request.method, request.url.path)
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "request_finished id=%s status=%s elapsed_ms=%.2f",
        request_id,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response
