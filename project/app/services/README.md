# Сервисы backend

В этой папке находится прикладная логика:

- загрузка артефактов модели;
- оркестрация скоринга (бизнес-правила → ML → штрафы → пороги);
- explainability (до 3 причин на русском).

Сервисы не зависят от HTTP-слоя и переиспользуются в тестах.

Реализовано:

- `inference_service.py`:
  - загрузка `model.joblib` и `feature_metadata.json`;
  - `apply_favorable_rules` / `apply_hard_rules` из `app/core/business_rules.py` (могут обойти ML);
  - расчёт `predict_proba`, post-ML штрафы, `resolve_decision`;
  - возврат `decision`, `probability`, `reasons` (1–3 строки).
- `explanation_service.py`:
  - SHAP `TreeExplainer` на ML-пути;
  - fallback по `feature_importance` из metadata.
