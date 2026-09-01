#!/usr/bin/env bash

set -u
set -o pipefail

section() {
    printf '\n## %s\n\n' "$1"
}

run_if_available() {
    local command_name="$1"
    shift

    if command -v "${command_name}" >/dev/null 2>&1; then
        printf '$'
        printf ' %q' "${command_name}" "$@"
        printf '\n'
        "${command_name}" "$@" 2>&1 || true
    else
        printf '%s: not found\n' "${command_name}"
    fi
}

print_file_if_readable() {
    local path="$1"

    if [[ -r "${path}" ]]; then
        printf '\n### %s\n\n' "${path}"
        sed -n '1,240p' "${path}"
    fi
}

printf '# Ascend Server Inventory\n'
printf '\nGenerated at: %s\n' "$(date --utc --iso-8601=seconds 2>/dev/null || date)"

section "Operating System"
run_if_available uname -a
print_file_if_readable /etc/os-release
run_if_available lscpu
run_if_available free -h
run_if_available df -h

section "Ascend Devices"
run_if_available npu-smi info
run_if_available npu-smi info -t board
run_if_available npu-smi info -t product
run_if_available npu-smi info -t usages

if command -v lspci >/dev/null 2>&1; then
    printf '$ lspci -nn\n'
    lspci -nn 2>&1 | grep -Ei 'Huawei|Ascend|Processing accelerators' || true
else
    printf 'lspci: not found\n'
fi

section "Ascend Driver and Firmware"
for path in \
    /usr/local/Ascend/driver/version.info \
    /usr/local/Ascend/driver/version.info.* \
    /etc/ascend_install.info; do
    for resolved_path in ${path}; do
        print_file_if_readable "${resolved_path}"
    done
done

section "CANN Installation"
printf 'ASCEND_HOME_PATH=%s\n' "${ASCEND_HOME_PATH:-<unset>}"
printf 'ASCEND_OPP_PATH=%s\n' "${ASCEND_OPP_PATH:-<unset>}"
printf 'ASCEND_AICPU_PATH=%s\n' "${ASCEND_AICPU_PATH:-<unset>}"
printf 'LD_LIBRARY_PATH=%s\n' "${LD_LIBRARY_PATH:-<unset>}"

for path in \
    /usr/local/Ascend/ascend-toolkit/latest/version.cfg \
    /usr/local/Ascend/ascend-toolkit/latest/*-version.info \
    /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info \
    /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascend_toolkit_install.info \
    /etc/Ascend/ascend_cann_install.info; do
    for resolved_path in ${path}; do
        print_file_if_readable "${resolved_path}"
    done
done

section "Python Environment"
run_if_available python3 --version
run_if_available python3 -m pip --version

if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY' 2>&1 || true
import importlib.metadata
import platform
import sys

packages = (
    "torch",
    "torch-npu",
    "vllm",
    "vllm-ascend",
    "transformers",
    "accelerate",
    "safetensors",
)

print(f"executable: {sys.executable}")
print(f"python: {platform.python_version()}")
for package in packages:
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = "not installed"
    print(f"{package}: {version}")

try:
    import torch

    print(f"torch.version: {torch.__version__}")
except Exception as error:
    print(f"torch import failed: {error!r}")

try:
    import torch_npu

    print(f"torch_npu.version: {torch_npu.__version__}")
    print(f"torch_npu.device_count: {torch_npu.npu.device_count()}")
except Exception as error:
    print(f"torch_npu import failed: {error!r}")
PY
fi

section "Container Runtime"
run_if_available docker version
run_if_available docker info
run_if_available podman version

section "Network Interfaces"
run_if_available ip -brief link
run_if_available ip -brief address

section "Relevant Environment Variables"
env | grep -E '^(ASCEND|ATB|HCCL|NPU|VLLM|PYTORCH|TORCH|LD_LIBRARY_PATH|PATH)=' | sort || true

section "Loaded Ascend-Related Kernel Modules"
if command -v lsmod >/dev/null 2>&1; then
    lsmod | grep -Ei 'drv|devmm|hisi|huawei|ascend' || true
else
    printf 'lsmod: not found\n'
fi

section "Device Nodes"
if [[ -d /dev ]]; then
    find /dev -maxdepth 1 \( -name 'davinci*' -o -name 'devmm*' -o -name 'hisi_hdc*' \) -printf '%f\n' 2>/dev/null | sort || true
fi

