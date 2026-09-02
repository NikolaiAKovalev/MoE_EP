#!/usr/bin/env python3

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any


REQUIRED_MODULES = ("torch", "torch_npu", "vllm", "vllm_ascend")
PACKAGE_NAMES = ("torch", "torch-npu", "vllm", "vllm-ascend")
MODEL_CONFIG_FILES = ("config.json", "quant_config.json")


def package_version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def validate_runtime() -> bool:
    print("## Python Runtime Validation\n")

    for distribution_name in PACKAGE_NAMES:
        print(f"{distribution_name}: {package_version(distribution_name)}")

    loaded_modules: dict[str, Any] = {}
    succeeded = True

    for module_name in REQUIRED_MODULES:
        try:
            loaded_modules[module_name] = importlib.import_module(module_name)
            print(f"{module_name}_import: OK")
        except Exception as error:
            succeeded = False
            print(f"{module_name}_import: FAILED: {error!r}")

    torch = loaded_modules.get("torch")
    if torch is not None:
        try:
            npu_available = torch.npu.is_available()
            npu_count = torch.npu.device_count()
            print(f"npu_available: {npu_available}")
            print(f"npu_count: {npu_count}")
            for device_id in range(npu_count):
                device_name = torch.npu.get_device_name(device_id)
                print(f"device_{device_id}: {device_name}")

            if not npu_available or npu_count == 0:
                succeeded = False
        except Exception as error:
            succeeded = False
            print(f"npu_runtime_check: FAILED: {error!r}")

    try:
        from vllm_ascend._build_info import __device_type__

        print(f"vllm_ascend_device_type: {__device_type__}")
    except Exception as error:
        print(f"vllm_ascend_build_info: UNAVAILABLE: {error!r}")

    return succeeded


def print_model_configuration(model_path: Path) -> bool:
    print("\n## Model Configuration\n")
    print(f"model_path: {model_path}")

    if not model_path.is_dir():
        print("model_path_status: directory not found")
        return False

    succeeded = True
    for filename in MODEL_CONFIG_FILES:
        path = model_path / filename
        print(f"\n### {filename}\n")
        if not path.is_file():
            print("not found")
            if filename == "config.json":
                succeeded = False
            continue

        try:
            contents = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            succeeded = False
            print(f"read_failed: {error!r}")
            continue

        print(json.dumps(contents, indent=2, ensure_ascii=False, sort_keys=True))

    return succeeded


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Ascend Python runtime and optionally print model "
            "and quantization configuration files."
        )
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        type=Path,
        help="Path to the local model directory.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    runtime_succeeded = validate_runtime()

    if arguments.model_path is None:
        print("\n## Model Configuration\n")
        print("Skipped: no model path was provided.")
        model_succeeded = True
    else:
        model_succeeded = print_model_configuration(arguments.model_path)

    return 0 if runtime_succeeded and model_succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())

