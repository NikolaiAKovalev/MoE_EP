#!/usr/bin/env bash

set -euo pipefail

MODEL_PATH="${1:?Usage: bash scripts/run_qwen3_ep2.sh MODEL_PATH}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"

vllm serve "${MODEL_PATH}" \
    --served-model-name Qwen3 \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --quantization ascend \
    --dtype float16 \
    --host 0.0.0.0 \
    --port 3057 \
    --max-model-len 3096 \
    --max-num-seqs 8 \
    --enforce-eager \
    2>&1 | tee qwen3-ep2.log

