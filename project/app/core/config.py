"""Конфигурация приложения через переменные окружения."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    """Настройки backend-приложения."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    app_name: str = Field(default="Credit Scoring API", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    app_log_level: str = Field(default="INFO", alias="APP_LOG_LEVEL")

    artifacts_dir: Path = Field(default=Path("ml/artifacts"), alias="ARTIFACTS_DIR")
    model_file: str = Field(default="model.joblib", alias="MODEL_FILE")
    feature_metadata_file: str = Field(default="feature_metadata.json", alias="FEATURE_METADATA_FILE")

    threshold_approve: float = Field(default=0.35, alias="THRESHOLD_APPROVE")
    threshold_reject: float = Field(default=0.65, alias="THRESHOLD_REJECT")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Возвращает кэшированный объект настроек."""
    return Settings()
