# ML модуль

Папка `ml/` содержит пайплайн машинного обучения:

- загрузка и подготовка данных;
- обучение baseline и улучшенной модели;
- сравнение метрик;
- сохранение артефактов для backend.

Обучение выполняется офлайн, а API использует только сохранённые артефакты.

## Источники данных

| `--source` | Поведение |
|------------|-----------|
| `auto` (по умолчанию) | `ml/data/raw/cs-training.csv` → иначе mock |
| `give_me_credit` | только реальный датасет (ошибка без файла) |
| `mock` | синтетический генератор |
| `csv` | явный `--data-path` (GMC или унифицированный CSV) |

Подробности: `ml/data/raw/README.md`

## Как запускать

```bash
cd project

# 1. Положить cs-training.csv в ml/data/raw/ (или --download)
python -m ml.data.download_data

# 2. Обучение на Give Me Some Credit
python -m ml.training.train --source give_me_credit

# 3. EDA
python -m ml.training.eda --source give_me_credit

# Быстрый прогон на подвыборке
python -m ml.training.train --source give_me_credit --max-rows 10000
```

Без реального файла (mock):

```bash
python -m ml.training.train --source mock
```

## Пайплайн

1. `ml/data/loader.py` — загрузка и выбор источника.
2. `ml/data/preprocessing.py` — препроцессинг.
3. `ml/training/train.py` — обучение, метрики в `ml/artifacts/metrics.json` (поле `data_source`).
4. `ml/training/eda.py` — графики в `ml/artifacts/eda/`.
