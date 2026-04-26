# Сервисы backend

В этой папке находится прикладная логика:

- загрузка артефактов модели;
- инференс и расчёт вероятности дефолта (PD);
- explainability (топ причин решения);
- применение decision rule (`APPROVE / REVIEW / REJECT`).

Сервисы должны быть независимы от HTTP-слоя и переиспользуемы в тестах.

Реализовано:

- `inference_service.py`:
  - загрузка `model.joblib` и `feature_metadata.json`;
  - расчёт `predict_proba`;
  - применение decision rule;
  - возврат `decision`, `probability`, `reasons`.
- `explanation_service.py`:
  - приоритетно SHAP-объяснение;
  - fallback: объяснение по `feature_importance`.
