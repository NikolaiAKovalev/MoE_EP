# MoE Expert-Parallel Inference on Ascend 310P

## Project Goal

Run the `Qwen3-30B-A3B-Instruct-2507_w8a8_pertoken` Mixture-of-Experts model with vLLM in expert-parallel mode on a server equipped with six Huawei Ascend 310P accelerators.

After establishing a correct and reproducible baseline, optimize the inference stack to reduce end-to-end latency. The initial optimization focus should be vLLM and the Ascend runtime. Model-level changes may be considered later if runtime and communication optimizations are insufficient.

## Available Hardware

- One NVIDIA RTX A3000 laptop GPU for local development, lightweight experiments, client tooling, and tests with smaller models.
- One server with two NVIDIA V100 GPUs for intermediate validation of vLLM, MoE behavior, and distributed execution. Performance results from this system are not directly transferable to Ascend because its kernels, quantization support, and communication stack differ.
- One server with six Huawei Ascend 310P accelerators, a Kunpeng 920 CPU, and 1 TB of system memory. This is the target deployment and benchmarking platform.

## Intended Parallelism

The primary distributed execution strategy is expert parallelism. MoE experts should be distributed across the six Ascend devices, while routed tokens are exchanged between devices as required by the model's router.

Tensor parallelism or a hybrid tensor-parallel and expert-parallel configuration may be evaluated if required by memory capacity, runtime limitations, or measured performance.

## Primary Performance Objective

The main objective is to reduce end-to-end request latency without causing an unacceptable loss of throughput or model quality.

The benchmark suite should measure:

- Time to first token (TTFT).
- Time per output token (TPOT) or inter-token latency.
- Prefill latency.
- Decode latency.
- Complete request latency.
- Request and token throughput.
- Latency percentiles, including p50, p95, and p99.
- Model quality whenever an optimization changes model behavior or numerical precision.

## Initial Technical Questions

Before changing vLLM or the model, verify that the target software stack supports the complete configuration:

- The exact Qwen3 MoE architecture and checkpoint.
- The exact `w8a8_pertoken` weight and activation quantization format.
- Six Ascend 310P devices.
- Expert parallelism rather than tensor parallelism alone.
- Required MoE routing, dispatch, combine, and collective communication operations.
- Compatible versions of vLLM, the Ascend integration, CANN, drivers, firmware, and related libraries.
- Sufficient device memory and supported execution kernels.

## Proposed Work Sequence

1. Inventory the target server hardware and installed software.
2. Identify the exact model artifact, configuration, tokenizer, and quantization metadata.
3. Establish a minimal correct single-request execution path.
4. Scale execution to all six Ascend 310P devices using expert parallelism.
5. Create a reproducible baseline with fixed workloads and configuration.
6. Profile prefill, decode, MoE routing, communication, kernels, synchronization, and memory use.
7. Optimize the runtime and serving configuration.
8. Optimize expert placement, token dispatch, and inter-device communication.
9. Optimize kernels and KV-cache behavior where possible.
10. Consider model-level modifications only after runtime bottlenecks are understood.
11. Validate every material change for correctness, latency, throughput, memory use, and quality.

## Optimization Priorities

The initial priority order is:

1. vLLM scheduling, batching, prefill/decode configuration, and unnecessary synchronization.
2. Expert placement and inter-device communication cost.
3. MoE dispatch/combine kernels and collective communication.
4. KV-cache allocation and memory management.
5. Model-level techniques such as routing changes, expert pruning or merging, changing the number of active experts, alternative quantization, or speculative decoding.

Model-level techniques must include quality evaluation because they may alter the output distribution or reduce model capability.

## Expected Deliverables

- Reproducible environment and launch instructions.
- Versioned runtime and model configuration.
- A fixed benchmark workload and measurement procedure.
- A working six-device expert-parallel baseline on Ascend 310P.
- Profiling results identifying the dominant latency bottlenecks.
- Documented optimizations with before-and-after measurements.
- A final comparison covering TTFT, TPOT, end-to-end latency, throughput, memory use, and quality impact.

## Project Language Convention

Communication between the user and the assistant is in Russian. All repository content must be written in English, including source code, comments, documentation, Markdown files, configuration text, scripts, test names, and generated project artifacts.
