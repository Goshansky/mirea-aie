"""Конфигурация pytest для проекта eda-cli."""
from __future__ import annotations

import sys
from pathlib import Path

# Добавляем src/ в sys.path для импортов
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
