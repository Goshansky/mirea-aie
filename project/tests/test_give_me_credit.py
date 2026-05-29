"""Тесты преобразования Give Me Some Credit."""

import pandas as pd

from ml.data.give_me_credit import is_give_me_credit_format, transform_give_me_credit
from ml.data.constants import TARGET_COLUMN
from ml.data.loader import load_dataset


def test_is_give_me_credit_format() -> None:
    """Проверяет определение формата датасета."""
    assert is_give_me_credit_format(pd.DataFrame({"SeriousDlqin2yrs": [0, 1]}))
    assert not is_give_me_credit_format(pd.DataFrame({"default": [0, 1]}))


def test_transform_give_me_credit_maps_features() -> None:
    """Проверяет маппинг колонок в признаки API."""
    raw = pd.DataFrame(
        {
            "SeriousDlqin2yrs": [0, 1],
            "age": [25, 45],
            "DebtRatio": [0.4, 1.2],
            "MonthlyIncome": [5000.0, 8000.0],
            "NumberOfOpenCreditLinesAndLoans": [3, 10],
            "NumberOfTimes90DaysLate": [0, 2],
            "NumberOfTime30-59DaysPastDueNotWorse": [1, 0],
            "NumberOfTime60-89DaysPastDueNotWorse": [0, 1],
        }
    )
    result = transform_give_me_credit(raw)

    expected_columns = {
        "income",
        "loan_amount",
        "age",
        "credit_history",
        "debt_ratio",
        "late_payments",
        TARGET_COLUMN,
    }
    assert expected_columns.issubset(set(result.columns))
    assert len(result) == 2
    assert result["late_payments"].iloc[1] == 3


def test_load_dataset_mock_source() -> None:
    """Проверяет загрузку mock-данных."""
    df, source = load_dataset(source="mock", random_state=42)
    assert source == "mock"
    assert len(df) > 0
    assert TARGET_COLUMN in df.columns
