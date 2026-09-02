#!/usr/bin/env python3

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RequestMetrics:
    ttft_ms: float
    e2e_ms: float
    tpot_ms: float | None
    output_tokens: int | None
    output_tokens_per_second: float | None


def percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def extract_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def run_request(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> RequestMetrics:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_completion_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    first_token_time: float | None = None
    completion_tokens: int | None = None

    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue

            data = line.removeprefix("data:").strip()
            if not data or data == "[DONE]":
                continue

            chunk = json.loads(data)
            if first_token_time is None and extract_text(chunk):
                first_token_time = time.perf_counter()

            usage = chunk.get("usage")
            if usage and usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])

    end = time.perf_counter()
    if first_token_time is None:
        raise RuntimeError("The response stream did not contain generated text.")

    ttft_seconds = first_token_time - start
    e2e_seconds = end - start
    decode_seconds = end - first_token_time

    if completion_tokens is not None and completion_tokens > 1:
        tpot_seconds = decode_seconds / (completion_tokens - 1)
    else:
        tpot_seconds = None

    if completion_tokens is not None and completion_tokens > 1 and decode_seconds > 0:
        output_tokens_per_second = (completion_tokens - 1) / decode_seconds
    else:
        output_tokens_per_second = None

    return RequestMetrics(
        ttft_ms=ttft_seconds * 1000,
        e2e_ms=e2e_seconds * 1000,
        tpot_ms=tpot_seconds * 1000 if tpot_seconds is not None else None,
        output_tokens=completion_tokens,
        output_tokens_per_second=output_tokens_per_second,
    )


def aggregate(measurements: list[RequestMetrics]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "request_count": len(measurements),
        "ttft_ms": summarize([item.ttft_ms for item in measurements]),
        "e2e_ms": summarize([item.e2e_ms for item in measurements]),
    }

    tpot_values = [item.tpot_ms for item in measurements if item.tpot_ms is not None]
    throughput_values = [
        item.output_tokens_per_second
        for item in measurements
        if item.output_tokens_per_second is not None
    ]
    if tpot_values:
        result["tpot_ms"] = summarize(tpot_values)
    if throughput_values:
        result["output_tokens_per_second"] = summarize(throughput_values)

    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure streaming latency of an OpenAI-compatible chat endpoint."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:3057/v1/chat/completions",
        help="Chat Completions endpoint URL.",
    )
    parser.add_argument("--model", default="Qwen3", help="Served model name.")
    parser.add_argument(
        "--prompt",
        default="Explain expert parallelism in exactly four short sentences.",
        help="Prompt used for every measured request.",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    arguments = parser.parse_args()

    if arguments.max_tokens < 2:
        parser.error("--max-tokens must be at least 2 for TPOT measurement.")
    if arguments.warmup < 0:
        parser.error("--warmup cannot be negative.")
    if arguments.requests < 1:
        parser.error("--requests must be positive.")
    return arguments


def main() -> int:
    arguments = parse_arguments()

    for index in range(arguments.warmup):
        print(f"Warm-up {index + 1}/{arguments.warmup}", flush=True)
        run_request(
            arguments.url,
            arguments.model,
            arguments.prompt,
            arguments.max_tokens,
            arguments.timeout,
        )

    measurements: list[RequestMetrics] = []
    for index in range(arguments.requests):
        metrics = run_request(
            arguments.url,
            arguments.model,
            arguments.prompt,
            arguments.max_tokens,
            arguments.timeout,
        )
        measurements.append(metrics)
        message = (
            f"Request {index + 1}/{arguments.requests}: "
            f"TTFT={metrics.ttft_ms:.2f} ms, E2E={metrics.e2e_ms:.2f} ms"
        )
        if metrics.tpot_ms is not None:
            message += f", TPOT={metrics.tpot_ms:.2f} ms"
        print(message, flush=True)

    report = {
        "configuration": {
            "url": arguments.url,
            "model": arguments.model,
            "prompt": arguments.prompt,
            "max_tokens": arguments.max_tokens,
            "warmup_requests": arguments.warmup,
            "measured_requests": arguments.requests,
            "concurrency": 1,
        },
        "summary": aggregate(measurements),
        "requests": [asdict(item) for item in measurements],
    }
    serialized_report = json.dumps(report, indent=2, ensure_ascii=False)
    print("\n" + serialized_report)

    if arguments.output is not None:
        arguments.output.write_text(serialized_report + "\n", encoding="utf-8")
        print(f"\nReport written to {arguments.output}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise SystemExit(f"Request failed: {error.reason}") from error
