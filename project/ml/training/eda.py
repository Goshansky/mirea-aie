"""Краткий EDA-скрипт для датасета кредитного скоринга."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ml.data.constants import TARGET_COLUMN
from ml.data.loader import load_dataset


def parse_args() -> argparse.Namespace:
    """Парсит аргументы запуска EDA."""
    parser = argparse.ArgumentParser(description="Краткий EDA для кредитного скоринга.")
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
        "--output-dir",
        type=str,
        default="ml/artifacts/eda",
        help="Папка для сохранения графиков и сводки.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Инициализация генератора.")
    return parser.parse_args()


def main() -> None:
    """Запускает разведочный анализ и сохраняет артефакты."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.data_path) if args.data_path else None
    df, data_source = load_dataset(
        dataset_path=dataset_path,
        random_state=args.random_state,
        source=args.source,  # type: ignore[arg-type]
    )
    print(f"Источник данных: {data_source}, строк: {len(df)}")

    summary = {
        "data_source": data_source,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "missing_values": df.isna().sum().to_dict(),
        "target_distribution": df[TARGET_COLUMN].value_counts(normalize=True).to_dict(),
    }
    summary_df = pd.DataFrame(
        {
            "feature": list(summary["missing_values"].keys()),
            "missing_count": list(summary["missing_values"].values()),
        }
    )
    summary_df.to_csv(output_dir / "missing_values.csv", index=False)
    pd.DataFrame([summary]).to_json(output_dir / "summary.json", orient="records", force_ascii=False, indent=2)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET_COLUMN)
    plt.title("Распределение целевого признака")
    plt.tight_layout()
    plt.savefig(output_dir / "target_distribution.png", dpi=140)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(df["debt_ratio"], kde=True, bins=30)
    plt.title("Распределение debt_ratio")
    plt.tight_layout()
    plt.savefig(output_dir / "debt_ratio_hist.png", dpi=140)
    plt.close()

    numeric_df = df.select_dtypes(include=["number"])
    correlation = numeric_df.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=False, cmap="coolwarm", center=0)
    plt.title("Корреляционная матрица числовых признаков")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=140)
    plt.close()

    print("EDA завершён. Файлы сохранены в:", output_dir)


if __name__ == "__main__":
    main()
