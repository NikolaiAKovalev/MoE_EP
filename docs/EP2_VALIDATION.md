# EP2 Validation

## Launch

Stop the TP1 baseline server, then run:

```bash
bash scripts/run_qwen3_ep2.sh /path/to/model
```

The script uses physical NPU devices `0,1` by default. Override the native Ascend visibility variable when another pair is required:

```bash
ASCEND_RT_VISIBLE_DEVICES=2,3 \
bash scripts/run_qwen3_ep2.sh /path/to/model
```

This configuration uses TP2 for attention and EP2 for MoE layers. It is not a pure EP-only configuration.

## Verify Expert Parallelism

After startup, inspect the recorded evidence:

```bash
bash scripts/check_qwen3_ep2.sh
```

Expected evidence includes:

- World size 2.
- EP ranks 0 and 1.
- Expert parallelism enabled on both ranks.
- 128 global experts and 64 local experts per rank.
- Different local-to-global expert maps for the two ranks.

A successful API request is also required to confirm that distributed token dispatch and expert result combination execute correctly.

