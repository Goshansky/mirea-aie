# Тесты проекта

Интеграционные и модульные тесты (`pytest`).

## Запуск

```bash
cd project
python -m pytest tests -v
```

Сейчас **22 теста**.

## Файлы

| Файл | Что проверяет |
|------|----------------|
| `test_health.py` | `GET /health` |
| `test_predict.py` | контракт `/predict` (ML, auto-approve, hard reject), валидация 422 |
| `test_business_rules.py` | авто-одобрение/отказ, штрафы к PD (возраст, линии, кредит) |
| `test_give_me_credit.py` | маппинг GMC, загрузка mock-датасета |

Подробности запуска — в `project/README.md` §8.
