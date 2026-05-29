# Backend-приложение (FastAPI)

В этой папке размещается **backend-часть проекта** для скоринга кредитных заявок.

Папка `app/` отвечает за:

- HTTP API (`/health`, `/predict`);
- валидацию входных данных и контрактов ответа;
- загрузку обученной модели и запуск инференса;
- применение бизнес-правил принятия решения (`APPROVE / REJECT / REVIEW`);
- формирование объяснений по решению.

Рекомендуемая структура:

- `api/` — роуты и зависимости API;
- `core/` — конфигурация, логирование, decision rule, бизнес-правила;
- `models/` — pydantic-схемы и типы инференса;
- `services/` — инференс и explainability;
- `main.py` — точка входа FastAPI-приложения и middleware логирования.

Запуск сервиса:

```bash
cd project
uvicorn app.main:app --reload --port 8000
```

Проверка endpoints:

```bash
curl http://127.0.0.1:8000/health
```

Ключевые требования к коду в `app/`:

- использовать type hints;
- не хардкодить секреты и пороги (использовать `.env`);
- логировать запросы и результаты предсказаний;
- обрабатывать ошибки через `HTTPException` с понятным `detail`.

Интерфейс пользователя реализуется отдельно в `project/frontend/` на React и обращается к этому API.

Детальные команды запуска и сценарии использования описывайте в `project/README.md`.

## Как это связано с ML

- Backend не обучает модель.
- Backend читает готовые артефакты из `ml/artifacts/`.
- После каждого переобучения (`python -m ml.training.train`) API начинает использовать новую финальную модель (`model.joblib`).

## Реализованные файлы

- `core/config.py` — настройки через `.env` (`THRESHOLD_APPROVE`, `THRESHOLD_REJECT`, пути артефактов).
- `core/logging.py` — базовая конфигурация логов.
- `core/decision.py` — пороги PD → `APPROVE / REVIEW / REJECT`.
- `core/business_rules.py` — авто-одобрение/отказ и штрафы к PD.
- `api/routes/health.py` — `GET /health`.
- `api/routes/predict.py` — `POST /predict` с обработкой ошибок.
- `services/inference_service.py` — загрузка `model.joblib`, правила, ML, PD, решение.
- `services/explanation_service.py` — SHAP + fallback по feature importance.
