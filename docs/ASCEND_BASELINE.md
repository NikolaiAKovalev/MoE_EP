# Ascend 310P Baseline Preparation

## Purpose

The first project milestone is to establish the exact hardware and software configuration of the target six-device Ascend 310P server. Version selection and deployment decisions must be based on this inventory rather than assumptions.

## Collect the Inventory

Copy the repository to the target server and run:

```bash
bash scripts/collect_ascend_inventory.sh | tee ascend-inventory.txt
```

The script performs read-only checks. It does not install packages, start services, or change system configuration.

Review the output before sharing it. In particular, check whether environment variables or network interface data contain information that should remain private.

## Validate the Container and Model

Run the focused runtime validation inside the target container:

```bash
bash scripts/validate_target_environment.sh
```

If the model is already available inside the container, pass its local path:

```bash
bash scripts/validate_target_environment.sh /path/to/model
```

The script validates Python package imports, torch-npu device visibility, the vLLM Ascend build target when available, and the model's configuration and quantization metadata.

## Run the Minimal Baseline

The baseline script intentionally contains only the model command and one device-selection variable:

```bash
bash scripts/run_qwen3_baseline.sh /path/to/model
```

It uses physical device 0 by default. Select another device through the native Ascend environment variable:

```bash
ASCEND_RT_VISIBLE_DEVICES=2 \
bash scripts/run_qwen3_baseline.sh /path/to/model
```

The Ascend quantization loader rejects FP32 and supports INT8, FP16, and BF16. The minimal baseline therefore uses FP16. Capture logs externally when needed:

```bash
bash scripts/run_qwen3_baseline.sh /path/to/model 2>&1 | tee qwen3-baseline.log
```

The `--enable-expert-parallel` flag alone does not prove that experts are distributed across multiple NPUs. With TP1 and DP1, the model normally has a distributed world size of one. Runtime logs or process/device utilization must demonstrate multiple ranks before this run can be classified as an expert-parallel deployment.

## Required Model Information

Record the following information separately for the exact checkpoint:

- Full local path or model registry identifier.
- Source and revision or commit hash.
- Model configuration file.
- Quantization configuration and implementation name.
- Weight file format and total size.
- Tokenizer revision.
- Whether custom model code is required.
- License and redistribution restrictions.

Do not add model weights, credentials, access tokens, or private registry URLs to Git.

## Baseline Acceptance Criteria

The initial baseline is complete when all of the following are true:

- All six Ascend 310P devices are visible and healthy.
- Driver, firmware, CANN, Python, PyTorch, and torch-npu versions are recorded.
- A compatible vLLM and Ascend integration combination is selected and pinned.
- The exact model artifact and quantization format are identified.
- The model loads without unsupported-operator or out-of-memory failures.
- A deterministic smoke-test request completes successfully.
- Expert parallelism is confirmed from configuration and runtime evidence.
- Launch commands, environment variables, logs, and observed memory use are recorded.

## Next Decision

After collecting the inventory, determine whether the existing server environment can support the model directly or whether a pinned container image or a separate Python environment is required. Do not upgrade the server stack before completing this compatibility assessment.
