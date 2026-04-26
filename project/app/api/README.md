# API слой

В этой папке находится HTTP-слой backend-приложения.

- `routes/` — маршруты (`/health`, `/predict`);
- `deps.py` — DI-фабрика `InferenceService`;
- валидация и сериализация выполняются через схемы из `app/models/`.

Задача слоя API: принять запрос, провалидировать вход, вызвать сервисы и вернуть стандартизированный ответ.

Текущий контракт:

- `GET /health` -> `{"status": "ok"}`;
- `POST /predict` -> `{"decision": "...", "probability": 0.xxxx, "reasons": [...]}`.
