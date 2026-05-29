"""Загрузка и генерация данных для кредитного скоринга."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data.constants import TARGET_COLUMN
from ml.data.give_me_credit import (
    RAW_DIR,
    find_give_me_credit_file,
    is_give_me_credit_format,
    load_give_me_credit_dataset,
    transform_give_me_credit,
)

logger = logging.getLogger(__name__)
DataSource = Literal["auto", "mock", "give_me_credit", "csv"]


def _generate_mock_dataset(n_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """Генерирует синтетический датасет кредитных заявок."""
    rng = np.random.default_rng(random_state)

    income = rng.normal(loc=120_000, scale=45_000, size=n_samples).clip(20_000, 400_000)
    loan_amount = rng.normal(loc=60_000, scale=25_000, size=n_samples).clip(5_000, 250_000)
    age = rng.normal(loc=39, scale=11, size=n_samples).clip(18, 75)
    credit_history = rng.integers(low=0, high=11, size=n_samples)
    debt_ratio = rng.beta(2.2, 5.5, size=n_samples).clip(0.01, 0.95)
    late_payments = rng.poisson(lam=1.2, size=n_samples).clip(0, 15)

    debt_to_income = loan_amount / income

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

    missing_fraction = 0.03
    for column in ["income", "loan_amount", "debt_ratio", "credit_history"]:
        mask = rng.random(n_samples) < missing_fraction
        df.loc[mask, column] = np.nan

    return df


def _load_from_csv(path: Path) -> tuple[pd.DataFrame, str]:
    """Загружает CSV: Give Me Some Credit или уже унифицированный формат."""
    raw = pd.read_csv(path)
    if is_give_me_credit_format(raw):
        return transform_give_me_credit(raw), "give_me_credit"
    if TARGET_COLUMN not in raw.columns:
        raise ValueError(
            f"В датасете нет '{TARGET_COLUMN}' и это не формат Give Me Some Credit "
            f"(ожидается колонка SeriousDlqin2yrs)."
        )
    return raw, "csv"


def load_dataset(
    dataset_path: Path | None = None,
    random_state: int = 42,
    source: DataSource = "auto",
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Загружает датасет и возвращает (DataFrame, имя_источника).

    source:
    - auto: Give Me Some Credit из raw/ -> иначе mock;
    - give_me_credit: только реальный датасет (ошибка, если файла нет);
    - mock: синтетика;
    - csv: явный путь через dataset_path.
    """
    resolved_source = source
    df: pd.DataFrame

    if source == "mock":
        df = _generate_mock_dataset(random_state=random_state)
    elif source == "give_me_credit":
        path = dataset_path or find_give_me_credit_file()
        df = load_give_me_credit_dataset(path)
    elif source == "csv":
        if dataset_path is None or not dataset_path.exists():
            raise FileNotFoundError("Для source=csv укажите существующий --data-path.")
        df, resolved_source = _load_from_csv(dataset_path)
    elif source == "auto":
        if dataset_path is not None and dataset_path.exists():
            df, resolved_source = _load_from_csv(dataset_path)
        else:
            gmc_path = find_give_me_credit_file()
            if gmc_path is not None:
                df = load_give_me_credit_dataset(gmc_path)
                resolved_source = "give_me_credit"
            else:
                df = _generate_mock_dataset(random_state=random_state)
                resolved_source = "mock"
                logger.warning(
                    "Give Me Some Credit не найден в %s. Используется mock. См. ml/data/raw/README.md",
                    RAW_DIR,
                )
    else:
        raise ValueError(f"Неизвестный source: {source}")

    if max_rows is not None and len(df) > max_rows:
        df, _ = train_test_split(
            df,
            train_size=max_rows,
            random_state=random_state,
            stratify=df[TARGET_COLUMN],
        )
        df = df.reset_index(drop=True)

    logger.info("Загружен датасет: source=%s, rows=%d", resolved_source, len(df))
    return df, resolved_source
