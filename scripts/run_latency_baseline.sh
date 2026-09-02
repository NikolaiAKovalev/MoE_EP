#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/benchmark_latency.py" \
    --url "${BENCHMARK_URL:-http://127.0.0.1:3057/v1/chat/completions}" \
    --model "${SERVED_MODEL_NAME:-Qwen3}" \
    --warmup "${WARMUP_REQUESTS:-2}" \
    --requests "${MEASURED_REQUESTS:-10}" \
    --max-tokens "${MAX_OUTPUT_TOKENS:-128}" \
    --output "${BENCHMARK_OUTPUT:-baseline-tp1-fp16.json}"

