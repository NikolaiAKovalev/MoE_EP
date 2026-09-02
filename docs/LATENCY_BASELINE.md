# Latency Baseline

## Objective

Record a reproducible single-request latency baseline before changing parallelism, kernels, scheduling, or model configuration.

## Run

Start the model server, then run from the server container or any client with network access to the endpoint:

```bash
bash scripts/run_latency_baseline.sh
```

The wrapper uses the documented baseline parameters. Override them through environment variables when running a separate experiment:

```bash
BENCHMARK_URL=http://127.0.0.1:3057/v1/chat/completions \
MEASURED_REQUESTS=20 \
MAX_OUTPUT_TOKENS=256 \
BENCHMARK_OUTPUT=baseline-tp1-fp16-256.json \
bash scripts/run_latency_baseline.sh
```

The script uses streaming responses and reports:

- Time to first token (TTFT).
- End-to-end request latency (E2E).
- Time per output token (TPOT).
- Output token throughput.
- Mean, minimum, maximum, p50, p95, and p99 values.

The benchmark is sequential and has concurrency 1. Warm-up requests are not included in the reported metrics.

## Measurement Rules

- Keep the prompt, output-token limit, server parameters, and request count fixed when comparing configurations.
- Run the client on the same machine for runtime-focused measurements.
- If the client runs remotely, network latency becomes part of E2E and TTFT.
- Do not compare the first cold request with warmed-up requests.
- Record server logs, NPU memory use, and NPU utilization alongside the latency report.
- Use a new output filename for every configuration.
