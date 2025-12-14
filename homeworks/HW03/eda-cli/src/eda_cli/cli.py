from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


def _create_json_summary(
    summary: DatasetSummary,
    quality_flags: Dict[str, Any],
    missing_df: pd.DataFrame,
    min_missing_share: float,
) -> Dict[str, Any]:
    """
    Создаёт компактную JSON-сводку по датасету.
    """
    # Собираем проблемные колонки
    problematic_columns = []
    
    # Колонки с пропусками выше порога
    if not missing_df.empty:
        problematic_missing = missing_df[missing_df["missing_share"] >= min_missing_share]
        for col_name in problematic_missing.index:
            problematic_columns.append({
                "name": col_name,
                "issue": "high_missing_share",
                "missing_share": float(missing_df.loc[col_name, "missing_share"]),
            })
    
    # Константные колонки
    if quality_flags.get("has_constant_columns", False):
        for col_name in quality_flags.get("constant_columns", []):
            problematic_columns.append({
                "name": col_name,
                "issue": "constant_column",
            })
    
    # Колонки с высокой кардинальностью
    if quality_flags.get("has_high_cardinality_categoricals", False):
        for col_name in quality_flags.get("high_cardinality_columns", []):
            problematic_columns.append({
                "name": col_name,
                "issue": "high_cardinality",
            })
    
    # Колонки с большим количеством нулей
    if quality_flags.get("has_many_zero_values", False):
        for col_name in quality_flags.get("zero_columns", []):
            problematic_columns.append({
                "name": col_name,
                "issue": "many_zero_values",
            })
    
    return {
        "n_rows": summary.n_rows,
        "n_cols": summary.n_cols,
        "quality_score": float(quality_flags.get("quality_score", 0.0)),
        "problematic_columns": problematic_columns,
        "quality_flags": {
            "too_few_rows": quality_flags.get("too_few_rows", False),
            "too_many_columns": quality_flags.get("too_many_columns", False),
            "too_many_missing": quality_flags.get("too_many_missing", False),
            "has_constant_columns": quality_flags.get("has_constant_columns", False),
            "has_high_cardinality_categoricals": quality_flags.get("has_high_cardinality_categoricals", False),
            "has_many_zero_values": quality_flags.get("has_many_zero_values", False),
        },
    }


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),
    top_k_categories: int = typer.Option(5, help="Сколько top-значений выводить для категориальных признаков."),
    title: str = typer.Option("EDA-отчёт", help="Заголовок отчёта."),
    min_missing_share: float = typer.Option(0.1, help="Порог доли пропусков, выше которого колонка считается проблемной."),
    json_summary: bool = typer.Option(False, "--json-summary", help="Сохранить компактную JSON-сводку по датасету."),
) -> None:
    """
    Сгенерировать полный EDA-отчёт:
    - текстовый overview и summary по колонкам (CSV/Markdown);
    - статистика пропусков;
    - корреляционная матрица;
    - top-k категорий по категориальным признакам;
    - картинки: гистограммы, матрица пропусков, heatmap корреляции.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Обзор
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, top_k=top_k_categories)

    # 2. Качество в целом
    quality_flags = compute_quality_flags(summary, missing_df, df)

    # 3. Сохраняем табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    # 4. Markdown-отчёт
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n")
        f.write(f"- Макс. доля пропусков по колонке: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Слишком мало строк: **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много колонок: **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Слишком много пропусков: **{quality_flags['too_many_missing']}**\n")
        
        # Новые эвристики
        f.write(f"- Есть константные колонки: **{quality_flags['has_constant_columns']}**\n")
        if quality_flags['has_constant_columns']:
            f.write(f"  - Константные колонки: {', '.join(quality_flags['constant_columns'])}\n")
        f.write(f"- Высокая кардинальность категориальных признаков: **{quality_flags['has_high_cardinality_categoricals']}**\n")
        if quality_flags['has_high_cardinality_categoricals']:
            f.write(f"  - Колонки с высокой кардинальностью: {', '.join(quality_flags['high_cardinality_columns'])}\n")
        f.write(f"- Много нулевых значений в числовых колонках: **{quality_flags['has_many_zero_values']}**\n")
        if quality_flags['has_many_zero_values']:
            f.write(f"  - Колонки с большим количеством нулей: {', '.join(quality_flags['zero_columns'])}\n")
        f.write(f"\n- Порог проблемных пропусков: **{min_missing_share:.1%}**\n\n")

        f.write("## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n\n")
            # Список проблемных колонок с пропусками выше порога
            problematic_missing = missing_df[missing_df["missing_share"] >= min_missing_share]
            if not problematic_missing.empty:
                f.write(f"### Проблемные колонки (пропусков ≥ {min_missing_share:.1%})\n\n")
                for col_name, row in problematic_missing.iterrows():
                    f.write(f"- `{col_name}`: {row['missing_share']:.1%} пропусков ({int(row['missing_count'])} из {summary.n_rows})\n")
                f.write("\n")

        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            f.write(f"См. файлы в папке `top_categories/` (топ-{top_k_categories} значений по каждой колонке).\n\n")

        f.write("## Гистограммы числовых колонок\n\n")
        f.write("См. файлы `hist_*.png`.\n\n")
        
        if json_summary:
            f.write("## JSON-сводка\n\n")
            f.write("Компактная сводка по датасету сохранена в файл `summary.json`.\n")

    # 5. JSON-сводка (если запрошена)
    if json_summary:
        json_summary_data = _create_json_summary(summary, quality_flags, missing_df, min_missing_share)
        json_path = out_root / "summary.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(json_summary_data, f, indent=2, ensure_ascii=False)
        typer.echo(f"- JSON-сводка: {json_path}")

    # 6. Картинки
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Основной markdown: {md_path}")
    typer.echo("- Табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    if json_summary:
        typer.echo("- JSON-сводка: summary.json")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png")


if __name__ == "__main__":
    app()
