# Target Environment

## Hardware

| Component | Observed value |
| --- | --- |
| CPU | Huawei Kunpeng 920 |
| System memory | 1 TB |
| Accelerator | 6 x Huawei Ascend 310P |
| Deployment form | Container |

The original project assumption of eight NPUs was corrected to six NPUs on 2026-09-01. All launch and parallelism configurations must use the observed device count.

## Container Software

| Component | Observed version |
| --- | --- |
| NPU SMI | `24.1.rc3.b050` |
| Ascend driver | `24.1.rc3.b050` |
| Ascend HAL | `7.35.23` |
| CANN | `9.1.0-beta.1` |
| Python | `3.12.13` |
| PyTorch | `2.10.0+cpu` |
| torch-npu | `2.10.0.post2` |
| vLLM | `0.23.0+empty` |
| vLLM Ascend | `0.23.0rc1` |
| Transformers | `5.5.4` |

These values were reported from inside the container. Host driver and firmware compatibility must be verified separately from the container user-space packages.

## Initial Compatibility Assessment

- Python 3.12 is within the documented vLLM Ascend range of Python 3.10 through 3.12.
- The documented inference-product stack pairs CANN `9.1.0-beta.1`, PyTorch `2.10.0`, and torch-npu `2.10.0.post2`.
- vLLM Ascend `0.23.0rc1` is aligned with upstream vLLM `0.23.0`.
- Ascend 310P support includes MoE and W8A8 capabilities, but these capabilities were introduced as experimental and must be validated for the exact checkpoint.
- The project's per-token activation quantization is expected to map to dynamic W8A8. This must be confirmed from the checkpoint's quantization configuration rather than inferred from its directory name.
- The exact combination of `Qwen3MoeForCausalLM`, dynamic W8A8, expert parallelism, and a world size of six is not yet considered validated.

The `+cpu` PyTorch build label is not by itself evidence that NPU execution is unavailable. NPU availability must be tested through torch-npu at runtime.

## Immediate Validation Gates

1. Confirm that all six devices are visible inside the container.
2. Confirm that `torch.npu.is_available()` is true and that torch-npu reports six devices.
3. Confirm the vLLM Ascend wheel build target is 310P.
4. Record the model's `config.json` and quantization metadata.
5. Verify that vLLM recognizes the model architecture and quantization method.
6. Load the model on one device if memory permits; otherwise perform the first load with tensor parallelism across all six devices.
7. Run a deterministic generation without expert parallelism to isolate basic model and quantization support.
8. Enable expert parallelism with the same workload and confirm it from runtime logs.
9. Compare output correctness, device memory, TTFT, and TPOT between the two runs.

## Parallelism Constraint

A six-device world size is legal for distributed execution, but it is less commonly represented in documented examples than powers of two. The official Qwen3-30B-A3B configuration has 128 routed experts with 8 experts activated per token. Because 128 is not divisible by 6, a uniform non-redundant placement is impossible: four ranks would hold 21 experts and two ranks would hold 22 experts. The exact vLLM Ascend path must support uneven or redundant expert placement; otherwise a different TP/EP layout or a reduced device set will be required.

## Primary References

- [vLLM Ascend installation guide](https://docs.vllm.ai/projects/ascend/en/main/getting_started/installation.html)
- [vLLM Ascend v0.23.0rc1 release notes](https://docs.vllm.ai/projects/ascend/en/v0.23.0rc1/user_guide/release_notes.html)
- [vLLM Ascend quantization guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html)
- [vLLM Ascend graph mode and Qwen3 MoE example](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/graph_mode.html)
- [Official Qwen3-30B-A3B-Instruct-2507 model configuration](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8/blob/main/config.json)
