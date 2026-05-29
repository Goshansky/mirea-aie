# Сырые данные Give Me Some Credit

Сюда кладётся **обучающий файл** соревнования Kaggle [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data):

- `cs-training.csv` — обучающая выборка
- `cs-testing.csv` — тестовая выборка Kaggle

Файлы **включены в репозиторий** (`project/ml/data/raw/`). После `git clone` отдельная загрузка не нужна.

## Как получить заново (если файлов нет)

### Вариант 1: вручную

1. Скачайте `cs-training.csv` с Kaggle.
2. Положите в эту папку: `project/ml/data/raw/cs-training.csv`.

### Вариант 2: Kaggle CLI

```bash
# Настройте ~/.kaggle/kaggle.json (API token)
cd project
python -m ml.data.download_data --download
```

### Проверка

```bash
cd project
python -m ml.data.download_data
```

## Обучение на реальных данных

```bash
cd project
python -m ml.training.train --source give_me_credit
python -m ml.training.eda --source give_me_credit
```
