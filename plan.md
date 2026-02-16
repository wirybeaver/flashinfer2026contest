# Plan: Track A FP8 MoE (Own Triton First, Then CUDA)

## Current Goal

Finish Track A in this order:
1. Triton own implementation and B200 tuning
2. CUDA implementation and B200 tuning

The local machine (RTX 4090) is for correctness checks. Modal B200 is the source of truth for performance.

## Constraints

- Keep contest scripts (`scripts/run_local.py`, `scripts/run_modal.py`) mostly unchanged.
- Keep implementation logic in `solution/*`.
- Avoid relying on hidden/internal FlashInfer fused-kernel shortcut as the final solution path.
- Use `uv` env (`source ~/fi-bench/bin/activate`) and no extra package installs.

## Progress Snapshot

- [x] Triton entrypoint implemented in `solution/triton/kernel.py` with Track A DPS signature.
- [x] `_ensure_shape` retained as commented reference for shape relationships.
- [x] One-off local verifier added (`scripts/verify_small_local.py`, git-excluded) to bypass full local harness OOM.
- [x] Local correctness verified on representative workloads:
  - `seq_len=7`, `32`, `901`, `11948`, `14107` (all pass, zero error).
- [ ] Modal B200 Triton benchmark pass with stable end-to-end timing output.
- [ ] Triton optimization loop based on B200 measurements.
- [ ] CUDA implementation and verification.

## Phase 1: Triton (Current)

1. Keep refining `solution/triton/kernel.py`:
   - DeepSeek no-aux routing (`top_k=8`, `n_group=8`, `topk_group=4`)
   - FP8 block-scale dequant
   - GEMM1 + SwiGLU + GEMM2 + weighted accumulation
2. Verify correctness locally with the tiny one-off verifier when full `run_local.py` is memory-blocked on 4090.
3. Run `modal run scripts/run_modal.py` for B200 data; debug timeout/runtime issues with minimal script disturbance.

## Phase 2: Triton B200 Optimization

1. Profile workload classes by `seq_len` on B200 (tiny/medium/long).
2. Improve kernel path where measurements show gains:
   - memory traffic/coalescing
   - routing + accumulation overhead
   - launch/sync overhead
3. Keep only changes that preserve correctness.

## Phase 3: CUDA

1. Implement `solution/cuda/binding.py` and `solution/cuda/kernel.cu`.
2. Keep semantics/signature identical to Triton implementation.
3. Verify locally first, then optimize on Modal B200.

## Validation and Delivery

- Keep both Triton and CUDA paths in repo.
- Compare B200 results and choose best submission path.
- Generate final `solution.json` from the chosen implementation.

## References

- Track A kernel page: https://bench.flashinfer.ai/kernels/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
- Dataset/setup: https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
- FP8 MoE API docs: https://docs.flashinfer.ai/api/fused_moe.html
- `trtllm_fp8_block_scale_moe`: https://docs.flashinfer.ai/generated/flashinfer.fused_moe.trtllm_fp8_block_scale_moe.html
- BYOK workflow: https://flashinfer-bench.mintlify.app/docs/tutorials/bring_your_own_kernel
- B200 architecture/tuning mindset reference: https://zcnrex.github.io/2025/12/23/nvfp4-gemm.html
