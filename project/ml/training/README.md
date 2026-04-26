# Обучение и оценка

В этой папке находятся скрипты:

- обучения baseline модели (логистическая регрессия);
- обучения улучшенной модели (например, RandomForest);
- расчёта метрик (`ROC-AUC`, `precision`, `recall`);
- сохранения результатов и сравнения моделей.

Итогом работы скриптов должны быть артефакты в `ml/artifacts/`.

## Текущие скрипты

- `train.py`:
  - обучает baseline и advanced модели;
  - сохраняет `model_baseline.joblib`, `model_advanced.joblib`, `model.joblib`;
  - сохраняет `metrics.json` и explainability metadata.
- `evaluate.py`:
  - считает метрики `ROC-AUC`, `precision`, `recall`.
- `explain.py`:
  - извлекает feature importance;
  - пытается посчитать SHAP summary (если доступен совместимый runtime).
- `eda.py`:
  - формирует краткий EDA в виде PNG/CSV/JSON.

## Запуск

```bash
cd project
python -m ml.training.train
python -m ml.training.eda
```
