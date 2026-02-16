# Future Runbook: Track A FP8 MoE

This file is a compact operational runbook for future sessions.

## Environment

- Activate env:
  - `source ~/fi-bench/bin/activate`
- Dataset:
  - `export FIB_DATASET_PATH=/workspace/mlsys26-contest`

## Safety Before Every Run

1. Ensure no stale GPU processes:
   - `nvidia-smi`
2. If a stale process exists and is known to be from a previous run, terminate it.
3. Re-check `nvidia-smi` before starting benchmark commands.

## Local Correctness Strategy

### Full harness (may OOM on 24GB GPUs)
- `python scripts/run_local.py`

### Lightweight one-off verifier
- `python scripts/verify_small_local.py`
- Optional selector:
  - `FIB_VERIFY_MAX_SEQ_LEN=32 python scripts/verify_small_local.py`

Known verified sequence lengths from one-off checks:
- `7`, `32`, `901`, `11948`, `14107` (all passed with zero error)

## Modal B200

- Baseline command:
  - `modal run scripts/run_modal.py`

If Modal times out or hangs on first run:
- Re-check logs/output and keep script edits minimal.
- Prefer staged reruns (smaller workload slices) only when needed for debugging.

## Implementation Boundaries

- Core kernel logic belongs in `solution/triton/kernel.py` and `solution/cuda/*`.
- Keep contest-provided scripts mostly unchanged.
- `_ensure_shape` lines in Triton kernel remain as commented shape references.

## Practical Tuning Priorities (B200)

1. Reduce routing/dequant overhead.
2. Improve memory traffic/coalescing.
3. Reduce launch/sync overhead.
4. Validate improvements with measured B200 results.

## Key References

- Track A: https://bench.flashinfer.ai/kernels/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
- FP8 MoE API: https://docs.flashinfer.ai/api/fused_moe.html
- `trtllm_fp8_block_scale_moe`: https://docs.flashinfer.ai/generated/flashinfer.fused_moe.trtllm_fp8_block_scale_moe.html
- BYOK: https://flashinfer-bench.mintlify.app/docs/tutorials/bring_your_own_kernel
- B200 tuning mindset note: https://zcnrex.github.io/2025/12/23/nvfp4-gemm.html
