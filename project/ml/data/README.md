# Данные и препроцессинг

В этой папке размещаются модули для:

- загрузки датасета (mock / Give Me Some Credit);
- базовой очистки данных;
- построения preprocessing pipeline (imputer/scaler/encoder).

## Модули

| Файл | Назначение |
|------|------------|
| `loader.py` | Единая точка загрузки (`auto`, `mock`, `give_me_credit`, `csv`) |
| `give_me_credit.py` | Чтение и маппинг Kaggle Give Me Some Credit |
| `download_data.py` | Проверка/скачивание `cs-training.csv` |
| `preprocessing.py` | `ColumnTransformer` для обучения и инференса |

## Give Me Some Credit

**Источник:** [Kaggle — Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data)  
**Файлы:** `raw/cs-training.csv`, `raw/cs-testing.csv` (в репозитории)

### Маппинг признаков → API

| Give Me Some Credit | Поле API | Логика |
|---------------------|----------|--------|
| `SeriousDlqin2yrs` | `default` | целевая переменная (дефолт) |
| `MonthlyIncome` | `income` | медиана для пропусков |
| `DebtRatio` × `MonthlyIncome` | `loan_amount` | оценка суммы обязательств |
| `DebtRatio` | `debt_ratio` | winsorize 99% → шкала [0, 1] |
| `age` | `age` | clip 18–100 |
| `NumberOfOpenCreditLinesAndLoans` | `credit_history` | прокси опыта |
| сумма колонок просрочек | `late_payments` | 90/30-59/60-89 дней |

## Команды

```bash
cd project

# Проверить наличие файла
python -m ml.data.download_data

# Скачать через Kaggle CLI (если настроен token)
python -m ml.data.download_data --download

# Обучение на реальных данных
python -m ml.training.train --source give_me_credit

# auto: GMC если есть в raw/, иначе mock
python -m ml.training.train --source auto
```

## Важно

- API и frontend по-прежнему принимают 6 полей — маппинг нужен только на этапе обучения.
- После переобучения на GMC перезапустите API, чтобы подхватить новые артефакты.
