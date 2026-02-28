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

Note:
- `verify_small_local.py` chooses the **first** workload with `seq_len <= FIB_VERIFY_MAX_SEQ_LEN`.
- To test an exact sequence length, use a targeted inline script (see below).

Known CUDA targeted checks completed:
- `seq_len=32`, `901`, `11948`, `14107` (status pass, parity still under refinement)

### Target exact `seq_len` (recommended for parity work)

Use this pattern to avoid ambiguity in workload selection:

```bash
TARGET_SEQ_LEN=14107 python - <<'PY'
import os, sys
from pathlib import Path
PROJECT_ROOT = Path('/workspace/flashinfer2026contest')
sys.path.insert(0, str(PROJECT_ROOT))
from flashinfer_bench import BenchmarkConfig, Solution, TraceSet
from flashinfer_bench.bench.evaluators import resolve_evaluator
from flashinfer_bench.compile import BuilderRegistry
from scripts.pack_solution import pack_solution

seq_len = int(os.environ["TARGET_SEQ_LEN"])
solution = Solution.model_validate_json(pack_solution().read_text())
trace = TraceSet.from_path(os.environ["FIB_DATASET_PATH"])
definition = trace.definitions[solution.definition]
workload = [
    wl for wl in trace.workloads[solution.definition]
    if int(wl.workload.axes.get("seq_len", -1)) == seq_len
][0].workload
cfg = BenchmarkConfig(
    warmup_runs=0, iterations=1, num_trials=1,
    sampling_validation_trials=1, profile_baseline=False
)
evaluator = resolve_evaluator(definition)
runnable = BuilderRegistry.get_instance().build(definition, solution)
baseline = evaluator.build_baseline(
    definition=definition, workload=workload, cfg=cfg,
    device="cuda:0", trace_set_root=trace.root
)
correctness, maybe_eval = evaluator.check_correctness(
    definition=definition, sol_runnable=runnable,
    inputs=baseline.inputs, ref_outputs=baseline.outputs,
    cfg=cfg, log_path="/tmp/fib_verify_target.log", device="cuda:0"
)
print("STATUS:", "PASSED" if maybe_eval is None else maybe_eval.status.value)
print(
    f"abs_err={correctness.max_absolute_error:.3e}, "
    f"rel_err={correctness.max_relative_error:.3e}"
)
PY
```

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
- CUDA entry is TVM-FFI in `solution/cuda/kernel.cu` with DPS style.
- `config.toml` for CUDA should stay:
  - `language = "cuda"`
  - `entry_point = "kernel.cu::kernel"`

## Current CUDA Status

- Device-side DeepSeek routing is implemented.
- Native FP8 conversion is used (`__nv_fp8_e4m3` path).
- Local targeted parity checks pass status, but long-seq parity still needs improvement.
- Full `run_local.py` can still OOM on 24GB 4090 due to baseline/reference memory pressure.

## Practical Tuning Priorities (B200)

1. Lock CUDA parity first (especially long-seq behavior).
2. Replace naive GEMM path with library-backed/fused path where safe.
3. Reduce routing/dequant overhead.
4. Improve memory traffic/coalescing.
5. Reduce launch/sync overhead.
6. Validate improvements with measured B200 results.

## Key References

- Track A: https://bench.flashinfer.ai/kernels/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
- FP8 MoE API: https://docs.flashinfer.ai/api/fused_moe.html
- `trtllm_fp8_block_scale_moe`: https://docs.flashinfer.ai/generated/flashinfer.fused_moe.trtllm_fp8_block_scale_moe.html
- BYOK: https://flashinfer-bench.mintlify.app/docs/tutorials/bring_your_own_kernel
- B200 tuning mindset note: https://zcnrex.github.io/2025/12/23/nvfp4-gemm.html
