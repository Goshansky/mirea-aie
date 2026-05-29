"""Проверка наличия и инструкции по загрузке Give Me Some Credit."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from ml.data.give_me_credit import DEFAULT_TRAIN_PATHS, RAW_DIR, find_give_me_credit_file

KAGGLE_COMPETITION = "GiveMeSomeCredit"
KAGGLE_FILE = "cs-training.csv"


def print_manual_instructions() -> None:
    """Печатает инструкцию ручной загрузки."""
    print("Give Me Some Credit (Kaggle)")
    print("1. Откройте: https://www.kaggle.com/c/GiveMeSomeCredit/data")
    print("2. Скачайте cs-training.csv")
    print(f"3. Сохраните файл в: {RAW_DIR}")
    print("4. Запустите обучение:")
    print("   cd project")
    print("   python -m ml.training.train --source give_me_credit")


def try_kaggle_download() -> bool:
    """Пытается скачать датасет через Kaggle CLI."""
    if shutil.which("kaggle") is None:
        print("Kaggle CLI не найден (команда `kaggle`).")
        return False

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        KAGGLE_COMPETITION,
        "-f",
        KAGGLE_FILE,
        "-p",
        str(RAW_DIR),
        "--force",
    ]
    print("Запуск:", " ".join(command))
    result = subprocess.run(command, check=False)
    return result.returncode == 0


def parse_args() -> argparse.Namespace:
    """Парсит аргументы CLI."""
    parser = argparse.ArgumentParser(description="Проверка/загрузка Give Me Some Credit.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Попытаться скачать через Kaggle CLI (нужны credentials).",
    )
    return parser.parse_args()


def main() -> None:
    """Проверяет наличие данных или запускает загрузку."""
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing = find_give_me_credit_file()
    if existing is not None:
        print(f"OK: найден файл {existing}")
        return

    print("Файл Give Me Some Credit не найден.")
    print("Ожидаемые имена:", ", ".join(DEFAULT_TRAIN_PATHS))
    print()

    if args.download:
        if try_kaggle_download():
            existing = find_give_me_credit_file()
            if existing is not None:
                print(f"OK: загружен {existing}")
                return
        print("Автозагрузка не удалась. Используйте ручную инструкцию ниже.")
        print()

    print_manual_instructions()
    sys.exit(1)


if __name__ == "__main__":
    main()
