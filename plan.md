# Plan: Track A FP8 MoE (Triton Done, CUDA Parity First)

## Current Goal

Finish Track A in this order:
1. Lock CUDA numerical parity on local 4090.
2. Move CUDA optimization loop to Modal B200.

Local 4090 is for correctness/parity. Modal B200 is the source of truth for performance.

## Constraints

- Keep contest scripts (`scripts/run_local.py`, `scripts/run_modal.py`) mostly unchanged.
- Keep implementation logic in `solution/*`.
- Keep CUDA path in TVM-FFI style (`kernel.cu::kernel`, DPS signature).
- Use `source ~/fi-bench/bin/activate`; no extra package installs.

## Progress Snapshot

- [x] Triton entrypoint implemented in `solution/triton/kernel.py`.
- [x] Triton local checks completed previously.
- [x] CUDA path switched to TVM-FFI entry (`config.toml`: `language="cuda"`, `entry_point="kernel.cu::kernel"`).
- [x] CUDA baseline implemented in `solution/cuda/kernel.cu`:
  - device-side DeepSeek routing
  - FP8 block-scale dequant
  - GEMM1 + SwiGLU + GEMM2 + weighted accumulation
- [x] CUDA local targeted checks pass status for `seq_len=32/901/11948/14107`.
- [ ] CUDA parity still not ideal (long-seq error remains elevated).
- [ ] CUDA B200 performance tuning not started.

## Current CUDA Parity Notes

- Routing parity is close to reference after:
  - native FP8 conversion (`__nv_fp8_e4m3`)
  - device-side grouped top-k routing
  - deterministic tie handling.
- Main remaining mismatch is likely in dequant/GEMM numerical path (not routing control flow).

## Next Actions (Priority Ordered)

1. **Parity stabilization**
   - Keep current routing path fixed.
   - Improve GEMM/dequant numerical alignment (avoid changes that regress long-seq stability).
2. **Introduce library-backed GEMM trial**
   - Replace one naive GEMM stage with a library path for parity A/B testing.
   - Keep routing and tensor plumbing unchanged to isolate effect.
3. **Regression protocol**
   - Validate on `seq_len=32, 901, 11948, 14107` after each significant change.
   - Record abs/rel error and keep only non-regressing changes.
4. **Then performance**
   - After parity lock, begin B200 throughput tuning.

## Validation and Delivery

- Keep both Triton and CUDA paths in repo.
- Compare CUDA/Triton B200 outcomes and choose submission path.
- Generate final `solution.json` from the chosen implementation.

## References

- Track A kernel page: https://bench.flashinfer.ai/kernels/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
- Dataset/setup: https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
- FP8 MoE API docs: https://docs.flashinfer.ai/api/fused_moe.html
- `trtllm_fp8_block_scale_moe`: https://docs.flashinfer.ai/generated/flashinfer.fused_moe.trtllm_fp8_block_scale_moe.html
- BYOK workflow: https://flashinfer-bench.mintlify.app/docs/tutorials/bring_your_own_kernel
- Starter-kit update (DPS/binding notes): https://github.com/flashinfer-ai/flashinfer-bench-starter-kit/commit/ef5b51a4b8ae6407397ba5e8e5e6a0f2f65430fe
