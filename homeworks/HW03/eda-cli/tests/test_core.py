from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df, df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_quality_flags_new_heuristics():
    """Тест новых эвристик качества данных."""
    # DataFrame с константной колонкой
    df_const = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "constant_col": [10, 10, 10, 10, 10],  # константная колонка
        "normal_col": [1, 2, 3, 4, 5],
    })
    
    summary = summarize_dataset(df_const)
    missing_df = missing_table(df_const)
    flags = compute_quality_flags(summary, missing_df, df_const)
    
    # Проверяем, что константная колонка обнаружена
    assert flags["has_constant_columns"] is True
    assert "constant_col" in flags["constant_columns"]
    
    # DataFrame с высокой кардинальностью
    df_high_card = pd.DataFrame({
        "id": list(range(100)),
        "high_card_col": [f"value_{i}" for i in range(100)],  # 100% уникальных значений
        "normal_col": ["A", "B"] * 50,
    })
    
    summary_high = summarize_dataset(df_high_card)
    missing_high = missing_table(df_high_card)
    flags_high = compute_quality_flags(summary_high, missing_high, df_high_card)
    
    # Проверяем, что высокая кардинальность обнаружена
    assert flags_high["has_high_cardinality_categoricals"] is True
    assert "high_card_col" in flags_high["high_cardinality_columns"]
    
    # DataFrame с большим количеством нулей
    df_zeros = pd.DataFrame({
        "id": list(range(10)),
        "zero_col": [0, 0, 0, 0, 0, 1, 2, 3, 4, 5],  # 50% нулей
        "normal_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })
    
    summary_zeros = summarize_dataset(df_zeros)
    missing_zeros = missing_table(df_zeros)
    flags_zeros = compute_quality_flags(summary_zeros, missing_zeros, df_zeros)
    
    # Проверяем, что много нулей обнаружено
    assert flags_zeros["has_many_zero_values"] is True
    assert "zero_col" in flags_zeros["zero_columns"]
