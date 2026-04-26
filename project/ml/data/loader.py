"""Загрузка и генерация данных для кредитного скоринга."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

TARGET_COLUMN: Final[str] = "default"


def _generate_mock_dataset(n_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """Генерирует синтетический датасет кредитных заявок."""
    rng = np.random.default_rng(random_state)

    income = rng.normal(loc=120_000, scale=45_000, size=n_samples).clip(20_000, 400_000)
    loan_amount = rng.normal(loc=60_000, scale=25_000, size=n_samples).clip(5_000, 250_000)
    age = rng.normal(loc=39, scale=11, size=n_samples).clip(18, 75)
    credit_history = rng.integers(low=0, high=11, size=n_samples)  # стаж кредитной истории, лет
    debt_ratio = rng.beta(2.2, 5.5, size=n_samples).clip(0.01, 0.95)
    late_payments = rng.poisson(lam=1.2, size=n_samples).clip(0, 15)

    debt_to_income = loan_amount / income

    # Логит-формула для вероятности дефолта.
    score = (
        -4.2
        + 2.6 * debt_ratio
        + 2.2 * debt_to_income
        + 0.17 * late_payments
        - 0.035 * credit_history
        - 0.012 * (age - 30)
    )
    probability_default = 1.0 / (1.0 + np.exp(-score))
    default = rng.binomial(1, p=probability_default, size=n_samples)

    df = pd.DataFrame(
        {
            "income": income.round(2),
            "loan_amount": loan_amount.round(2),
            "age": age.round(0).astype(int),
            "credit_history": credit_history.astype(int),
            "debt_ratio": debt_ratio.round(4),
            "late_payments": late_payments.astype(int),
            TARGET_COLUMN: default.astype(int),
        }
    )

    # Добавляем небольшую долю пропусков для проверки пайплайна.
    missing_fraction = 0.03
    for column in ["income", "loan_amount", "debt_ratio", "credit_history"]:
        mask = rng.random(n_samples) < missing_fraction
        df.loc[mask, column] = np.nan

    return df


def load_dataset(dataset_path: Path | None = None, random_state: int = 42) -> pd.DataFrame:
    """Загружает CSV-датасет или создаёт синтетический при отсутствии файла."""
    if dataset_path is not None and dataset_path.exists():
        data = pd.read_csv(dataset_path)
        if TARGET_COLUMN not in data.columns:
            raise ValueError(f"В датасете нет целевой колонки '{TARGET_COLUMN}'.")
        return data

    return _generate_mock_dataset(random_state=random_state)
