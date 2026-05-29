"""Производные признаки заявки для обучения и инференса."""

from __future__ import annotations

import pandas as pd

LOAN_TO_INCOME_COLUMN = "loan_to_income"


def enrich_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет loan_to_income = loan_amount / income.

    Отражает реальную нагрузку: сколько «месячных доходов» составляет сумма кредита.
    """
    result = df.copy()
    safe_income = result["income"].clip(lower=1)
    result[LOAN_TO_INCOME_COLUMN] = (result["loan_amount"] / safe_income).astype(float)
    return result
