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

printf '# Target Environment Validation\n\n'
printf 'Generated at: %s\n' "$(date --utc --iso-8601=seconds 2>/dev/null || date)"

printf '\n## Ascend Device Status\n\n'
if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info 2>&1 || true
else
    printf 'npu-smi: not found\n'
fi

printf '\n## Python Runtime Validation\n\n'
if ! command -v python3 >/dev/null 2>&1; then
    printf 'python3: not found\n' >&2
    exit 1
fi

python3 - <<'PY'
import importlib
import importlib.metadata


def package_version(distribution_name):
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


for distribution_name in ("torch", "torch-npu", "vllm", "vllm-ascend"):
    print(f"{distribution_name}: {package_version(distribution_name)}")

required_modules = ("torch", "torch_npu", "vllm", "vllm_ascend")
loaded_modules = {}
import_failed = False

for module_name in required_modules:
    try:
        loaded_modules[module_name] = importlib.import_module(module_name)
        print(f"{module_name}_import: OK")
    except Exception as error:
        import_failed = True
        print(f"{module_name}_import: FAILED: {error!r}")

torch = loaded_modules.get("torch")
if torch is not None:
    try:
        npu_available = torch.npu.is_available()
        npu_count = torch.npu.device_count()
        print(f"npu_available: {npu_available}")
        print(f"npu_count: {npu_count}")
        for device_id in range(npu_count):
            print(f"device_{device_id}: {torch.npu.get_device_name(device_id)}")
    except Exception as error:
        import_failed = True
        print(f"npu_runtime_check: FAILED: {error!r}")

try:
    from vllm_ascend._build_info import __device_type__

    print(f"vllm_ascend_device_type: {__device_type__}")
except Exception as error:
    print(f"vllm_ascend_build_info: UNAVAILABLE: {error!r}")

if import_failed:
    raise SystemExit(1)
PY
runtime_status=$?

if [[ -n "${model_path}" ]]; then
    printf '\n## Model Configuration\n\n'
    python3 - "${model_path}" <<'PY'
import json
import pathlib
import sys


model_path = pathlib.Path(sys.argv[1])
print(f"model_path: {model_path}")

if not model_path.is_dir():
    print("model_path_status: directory not found")
    raise SystemExit(1)

for filename in ("config.json", "quant_config.json"):
    path = model_path / filename
    print(f"\n### {filename}\n")
    if not path.is_file():
        print("not found")
        continue

    try:
        contents = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"read_failed: {error!r}")
        continue

    print(json.dumps(contents, indent=2, ensure_ascii=False, sort_keys=True))
PY
    model_status=$?
else
    printf '\n## Model Configuration\n\n'
    printf 'Skipped: no model path was provided.\n'
    model_status=0
fi

if (( runtime_status != 0 || model_status != 0 )); then
    exit 1
fi

