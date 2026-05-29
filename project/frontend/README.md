# Frontend (React)

Клиентское приложение на **React + TypeScript + Vite** для скоринга кредитных заявок.

## Функционал

- форма: `income`, `loan_amount`, `age`, `credit_history`, `debt_ratio`, `late_payments`;
- кнопка «Оценить» → `POST /predict`;
- вывод: решение (`APPROVE` / `REJECT` / `REVIEW`), вероятность дефолта, 1–3 причины на русском.

## Структура

- `src/App.tsx` — страница и состояние;
- `src/components/ScoringForm.tsx` — форма ввода;
- `src/components/ResultPanel.tsx` — блок результата;
- `src/api/client.ts` — HTTP-клиент к backend;
- `src/types/api.ts` — типы контракта API.

## Запуск

1. Поднять backend (из `project/`):

```bash
uvicorn app.main:app --reload --port 8000
```

2. Установить зависимости frontend:

```bash
cd project/frontend
npm install
```

3. Запустить dev-сервер:

```bash
npm run dev
```

Открыть: `http://localhost:5173`

В режиме dev запросы идут через proxy Vite: `/api/*` → `http://127.0.0.1:8000/*`.

## Production / Docker

В `docker compose` nginx на `:8080` проксирует `/api` → backend `:8000` — отдельный `VITE_API_URL` не нужен.

Локальная сборка:

```bash
npm run build
```

Прямой URL API (без proxy Vite) — в `frontend/.env.example`:

```env
VITE_API_URL=http://127.0.0.1:8000
```

## Что не коммитить

- `node_modules/`;
- `dist/`;
- локальный `.env` (если создаёте).
