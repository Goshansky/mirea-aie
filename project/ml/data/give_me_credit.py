"""Загрузка и преобразование датасета Give Me Some Credit (Kaggle)."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from ml.data.constants import TARGET_COLUMN

# Официальные имена колонок соревнования Kaggle «Give Me Some Credit».
COL_TARGET: Final[str] = "SeriousDlqin2yrs"
COL_AGE: Final[str] = "age"
COL_DEBT_RATIO: Final[str] = "DebtRatio"
COL_MONTHLY_INCOME: Final[str] = "MonthlyIncome"
COL_OPEN_LINES: Final[str] = "NumberOfOpenCreditLinesAndLoans"
COL_LATE_90: Final[str] = "NumberOfTimes90DaysLate"
COL_LATE_30_59: Final[str] = "NumberOfTime30-59DaysPastDueNotWorse"
COL_LATE_60_89: Final[str] = "NumberOfTime60-89DaysPastDueNotWorse"

RAW_DIR: Final[Path] = Path(__file__).resolve().parent / "raw"
DEFAULT_TRAIN_PATHS: Final[tuple[str, ...]] = (
    "cs-training.csv",
    "cs_train.csv",
    "CS_Training.csv",
)


def find_give_me_credit_file(explicit_path: Path | None = None) -> Path | None:
    """Ищет файл обучающей выборки Give Me Some Credit."""
    if explicit_path is not None:
        return explicit_path if explicit_path.exists() else None

    for name in DEFAULT_TRAIN_PATHS:
        candidate = RAW_DIR / name
        if candidate.exists():
            return candidate

    return None


def is_give_me_credit_format(df: pd.DataFrame) -> bool:
    """Проверяет, что DataFrame в формате Give Me Some Credit."""
    return COL_TARGET in df.columns


def load_give_me_credit_raw(path: Path) -> pd.DataFrame:
    """Читает сырой CSV Give Me Some Credit."""
    return pd.read_csv(path)


def transform_give_me_credit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит Give Me Some Credit к признакам API:
    income, loan_amount, age, credit_history, debt_ratio, late_payments, default.
    """
    required = [
        COL_TARGET,
        COL_AGE,
        COL_DEBT_RATIO,
        COL_MONTHLY_INCOME,
        COL_OPEN_LINES,
        COL_LATE_90,
        COL_LATE_30_59,
        COL_LATE_60_89,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"В датасете отсутствуют колонки: {missing}")

    work = df.copy()

    # Целевая переменная: дефолт за 2 года.
    work[TARGET_COLUMN] = work[COL_TARGET].astype(int)

    # Доход: пропуски заполняем медианой.
    income = pd.to_numeric(work[COL_MONTHLY_INCOME], errors="coerce")
    income_median = float(income.median())
    income = income.fillna(income_median).clip(lower=0)

    # Debt ratio: winsorize + нормализация в [0, 1] для согласованности с API.
    debt_ratio_raw = pd.to_numeric(work[COL_DEBT_RATIO], errors="coerce").replace([np.inf, -np.inf], np.nan)
    debt_ratio_raw = debt_ratio_raw.clip(lower=0)
    p99 = float(debt_ratio_raw.quantile(0.99))
    if p99 <= 0:
        p99 = 1.0
    debt_ratio = (debt_ratio_raw / p99).clip(0, 1).fillna(debt_ratio_raw.median())

    # Оценка суммы кредита: DebtRatio * MonthlyIncome (типичная эвристика для датасета).
    loan_amount = (debt_ratio_raw.fillna(0) * income).clip(lower=0)
    loan_median = float(loan_amount[loan_amount > 0].median()) if (loan_amount > 0).any() else 10_000.0
    loan_amount = loan_amount.mask(loan_amount <= 0, loan_median)

    age = pd.to_numeric(work[COL_AGE], errors="coerce").clip(18, 100).fillna(35).astype(int)
    credit_history = (
        pd.to_numeric(work[COL_OPEN_LINES], errors="coerce").fillna(0).clip(0, 80).astype(int)
    )

    late_90 = pd.to_numeric(work[COL_LATE_90], errors="coerce").fillna(0)
    late_30_59 = pd.to_numeric(work[COL_LATE_30_59], errors="coerce").fillna(0)
    late_60_89 = pd.to_numeric(work[COL_LATE_60_89], errors="coerce").fillna(0)
    late_payments = (late_90 + late_30_59 + late_60_89).clip(0, 50).astype(int)

    unified = pd.DataFrame(
        {
            "income": income.round(2),
            "loan_amount": loan_amount.round(2),
            "age": age,
            "credit_history": credit_history,
            "debt_ratio": debt_ratio.round(4),
            "late_payments": late_payments,
            TARGET_COLUMN: work[TARGET_COLUMN],
        }
    )

    unified = unified.dropna(subset=[TARGET_COLUMN])
    unified[TARGET_COLUMN] = unified[TARGET_COLUMN].astype(int)
    return unified.reset_index(drop=True)


def load_give_me_credit_dataset(path: Path | None = None) -> pd.DataFrame:
    """Загружает и преобразует Give Me Some Credit."""
    resolved = find_give_me_credit_file(path)
    if resolved is None:
        raise FileNotFoundError(
            "Не найден файл Give Me Some Credit. "
            f"Положите cs-training.csv в {RAW_DIR} или укажите --data-path."
        )

    raw = load_give_me_credit_raw(resolved)
    return transform_give_me_credit(raw)
