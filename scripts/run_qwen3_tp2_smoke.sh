#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_qwen3_tp2_smoke.sh MODEL_PATH

Optional environment variables:
  ASCEND_DEVICES       Visible NPU IDs. Default: 0,1
  VLLM_HOST            Server bind address. Default: 127.0.0.1
  VLLM_PORT            Server port. Default: 8000
  SERVED_MODEL_NAME    API model name. Default: qwen3
  STARTUP_LOG          Startup log path. Default: qwen3-tp2-startup.log
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
ascend_devices="${ASCEND_DEVICES:-0,1}"
vllm_host="${VLLM_HOST:-127.0.0.1}"
vllm_port="${VLLM_PORT:-8000}"
served_model_name="${SERVED_MODEL_NAME:-qwen3}"
startup_log="${STARTUP_LOG:-qwen3-tp2-startup.log}"

if [[ ! -d "${model_path}" ]]; then
    printf 'Model directory does not exist: %s\n' "${model_path}" >&2
    exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
    printf 'vllm executable was not found in PATH.\n' >&2
    exit 1
fi

device_count="$(awk -F, '{print NF}' <<<"${ascend_devices}")"
if [[ "${device_count}" != "2" ]]; then
    printf 'This TP2 smoke test requires exactly two NPU IDs; received: %s\n' \
        "${ascend_devices}" >&2
    exit 1
fi

export ASCEND_RT_VISIBLE_DEVICES="${ascend_devices}"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

printf 'Starting Qwen3 TP2 smoke test\n'
printf 'Model path: %s\n' "${model_path}"
printf 'Visible NPUs: %s\n' "${ASCEND_RT_VISIBLE_DEVICES}"
printf 'Endpoint: http://%s:%s\n' "${vllm_host}" "${vllm_port}"
printf 'Startup log: %s\n' "${startup_log}"

vllm serve "${model_path}" \
    --host "${vllm_host}" \
    --port "${vllm_port}" \
    --served-model-name "${served_model_name}" \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --distributed-executor-backend mp \
    --dtype float16 \
    --quantization ascend \
    --max-model-len 4096 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching \
    --additional-config \
        '{"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npu_graph_ex":false}}' \
    --compilation-config \
        '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}' \
    2>&1 | tee "${startup_log}"

