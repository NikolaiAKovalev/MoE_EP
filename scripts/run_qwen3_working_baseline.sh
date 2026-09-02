#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_qwen3_working_baseline.sh MODEL_PATH

Optional environment variables:
  ASCEND_DEVICES       Visible physical NPU IDs, for example: 2 or 2,3
                       If unset, the script preserves container visibility.
  VLLM_HOST            Server bind address. Default: 0.0.0.0
  VLLM_PORT            Server port. Default: 3057
  SERVED_MODEL_NAME    API model name. Default: Qwen3
  STARTUP_LOG          Startup log path. Default: qwen3-working-baseline.log
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if (( $# != 1 )); then
    usage >&2
    exit 2
fi

model_path="$1"
vllm_host="${VLLM_HOST:-0.0.0.0}"
vllm_port="${VLLM_PORT:-3057}"
served_model_name="${SERVED_MODEL_NAME:-Qwen3}"
startup_log="${STARTUP_LOG:-qwen3-working-baseline.log}"

if [[ ! -d "${model_path}" ]]; then
    printf 'Model directory does not exist: %s\n' "${model_path}" >&2
    exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
    printf 'vllm executable was not found in PATH.\n' >&2
    exit 1
fi

if [[ -n "${ASCEND_DEVICES:-}" ]]; then
    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_DEVICES}"
fi

printf 'Starting the known-working Qwen3 baseline\n'
printf 'Model path: %s\n' "${model_path}"
printf 'Visible NPUs: %s\n' "${ASCEND_RT_VISIBLE_DEVICES:-preserved from container}"
printf 'Endpoint: http://%s:%s\n' "${vllm_host}" "${vllm_port}"
printf 'Startup log: %s\n' "${startup_log}"

vllm serve "${model_path}" \
    --served-model-name "${served_model_name}" \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --quantization ascend \
    --dtype float32 \
    --host "${vllm_host}" \
    --port "${vllm_port}" \
    --max-model-len 3096 \
    --max-num-seqs 8 \
    --enforce-eager \
    2>&1 | tee "${startup_log}"

