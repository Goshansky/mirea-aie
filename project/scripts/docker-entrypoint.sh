#!/bin/sh
set -e

if [ ! -f "ml/artifacts/model.joblib" ]; then
  echo "[entrypoint] model.joblib не найден — запускаем быстрое обучение на mock-данных..."
  python -m ml.training.train --source mock --max-rows 5000
fi

exec "$@"
