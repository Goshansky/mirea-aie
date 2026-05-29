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

    threshold_approve: float = Field(default=0.25, alias="THRESHOLD_APPROVE")
    threshold_reject: float = Field(default=0.55, alias="THRESHOLD_REJECT")

    min_income: float = Field(default=15_000, alias="MIN_INCOME")
    max_loan_to_income: float = Field(default=24, alias="MAX_LOAN_TO_INCOME")
    auto_approve_max_loan_to_income: float = Field(default=3.0, alias="AUTO_APPROVE_MAX_LOAN_TO_INCOME")
    auto_approve_max_debt_ratio: float = Field(default=0.35, alias="AUTO_APPROVE_MAX_DEBT_RATIO")
    auto_approve_max_late_payments: int = Field(default=1, alias="AUTO_APPROVE_MAX_LATE_PAYMENTS")
    auto_approve_min_age: int = Field(default=21, alias="AUTO_APPROVE_MIN_AGE")
    auto_approve_max_age: int = Field(default=70, alias="AUTO_APPROVE_MAX_AGE")

    age_reject_min: int = Field(default=18, alias="AGE_REJECT_MIN")
    age_reject_max: int = Field(default=75, alias="AGE_REJECT_MAX")
    age_penalty_young_threshold: int = Field(default=23, alias="AGE_PENALTY_YOUNG_THRESHOLD")
    age_penalty_old_threshold: int = Field(default=65, alias="AGE_PENALTY_OLD_THRESHOLD")
    age_penalty_rate: float = Field(default=0.012, alias="AGE_PENALTY_RATE")
    age_penalty_cap: float = Field(default=0.40, alias="AGE_PENALTY_CAP")

    late_payments_reject: int = Field(default=3, alias="LATE_PAYMENTS_REJECT")
    late_payment_penalty: float = Field(default=0.06, alias="LATE_PAYMENT_PENALTY")
    late_payment_penalty_cap: float = Field(default=0.30, alias="LATE_PAYMENT_PENALTY_CAP")

    loan_penalty_free_threshold: float = Field(default=3.0, alias="LOAN_PENALTY_FREE_THRESHOLD")
    loan_penalty_rate: float = Field(default=0.02, alias="LOAN_PENALTY_RATE")
    loan_penalty_cap: float = Field(default=0.45, alias="LOAN_PENALTY_CAP")

    debt_penalty_start: float = Field(default=0.5, alias="DEBT_PENALTY_START")
    debt_penalty_rate: float = Field(default=0.25, alias="DEBT_PENALTY_RATE")
    debt_penalty_cap: float = Field(default=0.25, alias="DEBT_PENALTY_CAP")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Возвращает кэшированный объект настроек."""
    return Settings()
