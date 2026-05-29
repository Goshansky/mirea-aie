"""Препроцессинг признаков для обучения и инференса."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.data.constants import TARGET_COLUMN


@dataclass(frozen=True)
class FeatureConfig:
    """Конфигурация признаков для пайплайна."""

    numeric_features: Sequence[str]
    categorical_features: Sequence[str]

    @property
    def all_features(self) -> list[str]:
        """Возвращает полный список признаков для модели."""
        return list(self.numeric_features) + list(self.categorical_features)


def infer_feature_config(df: pd.DataFrame) -> FeatureConfig:
    """Выделяет числовые и категориальные признаки из датафрейма."""
    features = [column for column in df.columns if column != TARGET_COLUMN]
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(df[column])]
    categorical = [column for column in features if column not in numeric]
    return FeatureConfig(numeric_features=numeric, categorical_features=categorical)


def build_preprocessor(feature_config: FeatureConfig) -> ColumnTransformer:
    """Строит единый preprocessing pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(feature_config.numeric_features)),
            ("cat", categorical_pipeline, list(feature_config.categorical_features)),
        ]
    )
    return preprocessor
