#!/usr/bin/env bash

set -u
set -o pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/validate_target_environment.sh [MODEL_PATH]

The model path is optional. When provided, the script prints the model and
quantization configuration files in addition to validating the Ascend runtime.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if (( $# > 1 )); then
    usage >&2
    exit 2
fi

model_path="${1:-}"
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

printf '# Target Environment Validation\n\n'
printf 'Generated at: %s\n' "$(date --utc --iso-8601=seconds 2>/dev/null || date)"

printf '\n## Ascend Device Status\n\n'
if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info 2>&1 || true
else
    printf 'npu-smi: not found\n'
fi

if ! command -v python3 >/dev/null 2>&1; then
    printf 'python3: not found\n' >&2
    exit 1
fi

if [[ -n "${model_path}" ]]; then
    python3 "${script_directory}/validate_target_environment.py" "${model_path}"
else
    python3 "${script_directory}/validate_target_environment.py"
fi
