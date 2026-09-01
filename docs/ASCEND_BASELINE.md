# Ascend 310P Baseline Preparation

## Purpose

The first project milestone is to establish the exact hardware and software configuration of the target eight-device Ascend 310P server. Version selection and deployment decisions must be based on this inventory rather than assumptions.

## Collect the Inventory

Copy the repository to the target server and run:

```bash
bash scripts/collect_ascend_inventory.sh | tee ascend-inventory.txt
```

The script performs read-only checks. It does not install packages, start services, or change system configuration.

Review the output before sharing it. In particular, check whether environment variables or network interface data contain information that should remain private.

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

- All eight Ascend 310P devices are visible and healthy.
- Driver, firmware, CANN, Python, PyTorch, and torch-npu versions are recorded.
- A compatible vLLM and Ascend integration combination is selected and pinned.
- The exact model artifact and quantization format are identified.
- The model loads without unsupported-operator or out-of-memory failures.
- A deterministic smoke-test request completes successfully.
- Expert parallelism is confirmed from configuration and runtime evidence.
- Launch commands, environment variables, logs, and observed memory use are recorded.

## Next Decision

After collecting the inventory, determine whether the existing server environment can support the model directly or whether a pinned container image or a separate Python environment is required. Do not upgrade the server stack before completing this compatibility assessment.

