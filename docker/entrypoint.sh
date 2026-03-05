#!/usr/bin/env sh
set -eu

if [ "$#" -gt 0 ]; then
  export INFERENCE_APP_ARGS="$*"
fi

exec gunicorn src.inference:app \
  --bind 0.0.0.0:7860 \
  --workers 1 \
  --worker-class gthread \
  --threads 8 \
  --timeout 180
