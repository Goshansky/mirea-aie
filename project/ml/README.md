# ML модуль

Папка `ml/` содержит пайплайн машинного обучения:

- загрузка и подготовка данных;
- обучение baseline и улучшенной модели;
- сравнение метрик;
- сохранение артефактов для backend.

Обучение выполняется офлайн, а API использует только сохранённые артефакты.

## Как работает пайплайн

1. `ml/data/loader.py`:
   - если передан `--data-path`, читается CSV;
   - если файл не передан, генерируется синтетический датасет с целевой переменной `default`.
2. `ml/data/preprocessing.py`:
   - строится `ColumnTransformer` с обработкой пропусков;
   - для числовых признаков: `median + scaling`;
   - для категориальных: `most_frequent + one-hot`.
3. `ml/training/train.py`:
   - обучаются две модели: baseline (LogisticRegression) и advanced (RandomForest);
   - считаются `ROC-AUC`, `precision`, `recall`;
   - сохраняются артефакты в `ml/artifacts/`.
4. `ml/training/eda.py`:
   - сохраняет короткий EDA-отчёт (графики + сводные таблицы) в `ml/artifacts/eda/`.

## Как запускать

```bash
cd project
python -m ml.training.train
python -m ml.training.eda
```

Запуск с внешним датасетом:

```bash
cd project
python -m ml.training.train --data-path "C:/path/to/credit.csv"
python -m ml.training.eda --data-path "C:/path/to/credit.csv"
```
