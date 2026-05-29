"""Обучение baseline и улучшенной модели кредитного скоринга."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.data.constants import TARGET_COLUMN
from ml.data.loader import load_dataset
from ml.data.preprocessing import build_preprocessor, infer_feature_config
from ml.training.evaluate import calculate_metrics, save_metrics
from ml.training.explain import (
    calculate_shap_summary,
    get_feature_names,
    get_tree_feature_importance,
    save_feature_metadata,
)


def parse_args() -> argparse.Namespace:
    """Парсит аргументы запуска."""
    parser = argparse.ArgumentParser(description="Обучение моделей кредитного скоринга.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="",
        help="Путь к CSV (Give Me Some Credit или унифицированный формат).",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["auto", "mock", "give_me_credit", "csv"],
        default="auto",
        help="Источник данных: auto | mock | give_me_credit | csv.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Ограничить число строк (0 = без ограничения). Удобно для быстрых прогонов.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="ml/artifacts",
        help="Папка для сохранения артефактов.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Доля тестовой выборки.")
    parser.add_argument("--random-state", type=int, default=42, help="Инициализация генератора.")
    return parser.parse_args()


def _build_baseline_pipeline(preprocessor: object, random_state: int) -> Pipeline:
    """Создаёт baseline pipeline с логистической регрессией."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1200,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def _build_advanced_pipeline(preprocessor: object, random_state: int) -> Pipeline:
    """Создаёт улучшенный pipeline с RandomForest."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=10,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    """Основной сценарий обучения и сохранения артефактов."""
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.data_path) if args.data_path else None
    max_rows = args.max_rows if args.max_rows > 0 else None
    df, data_source = load_dataset(
        dataset_path=dataset_path,
        random_state=args.random_state,
        source=args.source,  # type: ignore[arg-type]
        max_rows=max_rows,
    )
    print(f"Источник данных: {data_source}, строк: {len(df)}")

    feature_config = infer_feature_config(df)
    preprocessor = build_preprocessor(feature_config)

    x = df[feature_config.all_features]
    y = df[TARGET_COLUMN].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    baseline_pipeline = _build_baseline_pipeline(preprocessor=preprocessor, random_state=args.random_state)
    baseline_pipeline.fit(x_train, y_train)
    baseline_proba = baseline_pipeline.predict_proba(x_test)[:, 1]
    baseline_metrics = calculate_metrics(y_test, baseline_proba)

    advanced_pipeline = _build_advanced_pipeline(preprocessor=preprocessor, random_state=args.random_state)
    advanced_pipeline.fit(x_train, y_train)
    advanced_proba = advanced_pipeline.predict_proba(x_test)[:, 1]
    advanced_metrics = calculate_metrics(y_test, advanced_proba)

    metrics_payload = {
        "data_source": data_source,
        "rows": len(df),
        "baseline_logistic_regression": baseline_metrics,
        "advanced_random_forest": advanced_metrics,
        "selected_model": "advanced_random_forest",
    }
    save_metrics(metrics_payload, artifacts_dir / "metrics.json")

    # Сохраняем обе модели и отдельно финальные артефакты для API.
    joblib.dump(baseline_pipeline, artifacts_dir / "model_baseline.joblib")
    joblib.dump(advanced_pipeline, artifacts_dir / "model_advanced.joblib")
    joblib.dump(advanced_pipeline, artifacts_dir / "model.joblib")

    fitted_preprocessor = advanced_pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(fitted_preprocessor)
    feature_importance = get_tree_feature_importance(advanced_pipeline, feature_names)

    transformed_sample = fitted_preprocessor.transform(x_test.iloc[: min(500, len(x_test))])
    if hasattr(transformed_sample, "toarray"):
        transformed_sample = transformed_sample.toarray()

    shap_summary = calculate_shap_summary(advanced_pipeline, transformed_sample, feature_names)
    save_feature_metadata(
        feature_names=feature_names,
        importances=feature_importance,
        output_path=artifacts_dir / "feature_metadata.json",
    )
    if shap_summary:
        save_feature_metadata(
            feature_names=feature_names,
            importances=shap_summary,
            output_path=artifacts_dir / "shap_summary.json",
        )

    np.save(artifacts_dir / "x_test_sample.npy", np.asarray(transformed_sample))
    print("Обучение завершено. Артефакты сохранены в:", artifacts_dir)


if __name__ == "__main__":
    main()
