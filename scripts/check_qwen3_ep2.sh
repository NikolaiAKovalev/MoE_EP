#!/usr/bin/env bash

set -euo pipefail

LOG_PATH="${1:-qwen3-ep2.log}"

grep -E \
    'world size|EP rank|EP Rank|Expert parallelism is enabled|Local/global number of experts|Experts local to global' \
    "${LOG_PATH}"

