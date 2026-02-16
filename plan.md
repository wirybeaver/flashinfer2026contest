# Plan: Complete Track A FP8 MoE (Triton first, then CUDA)

## To-Do List (Reset)

- [ ] Confirm `uv` env is active and set `FIB_DATASET_PATH=/workspace/mlsys26-contest`; verify local benchmark can load definition/workloads.
- [ ] Implement Track A-compatible `kernel(...)` in `solution/triton/kernel.py` using correctness-first FP8 MoE API path with strict input validation.
- [ ] Run `pack_solution.py` and `run_local.py` for Triton mode, fix interface/runtime issues, and capture baseline correctness/latency.
- [ ] Iterate Triton-path optimizations only where benchmark indicates meaningful wins while preserving correctness.
- [ ] Implement matching CUDA binding and kernel in `solution/cuda/binding.py` and `solution/cuda/kernel.cu` with same semantics/signature as Triton path.
- [ ] Run local benchmark in CUDA mode, fix issues, and optimize launch/memory behavior on RTX 4090.
- [ ] Compare Triton vs CUDA results, preserve both implementations, and prepare `solution.json` for chosen submission path.

## Scope and Constraints

- Target kernel: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`.
- Work in your existing `uv` environment (`source ~/fi-bench/bin/activate`), no package installs.
- Use local dataset at `/workspace/mlsys26-contest` and set `FIB_DATASET_PATH` accordingly for runs.

## Phase 1: Triton Path (Correctness Baseline First)

1. Replace template in `/workspace/flashinfer2026contest/solution/triton/kernel.py` with a Python entry function `kernel(...)` that matches the definition inputs exactly and returns the required `bfloat16` output.
2. Implement baseline logic by calling FlashInfer FP8 MoE API when available (same semantics as definition reference), with explicit shape/dtype checks and deterministic handling of scalar args (`local_expert_offset`, `routed_scaling_factor`).
3. Add lightweight fast-path/guard code to avoid accidental slow Python fallback on large `seq_len`.
4. Run local pack + benchmark loop and fix signature/runtime mismatches until all workloads complete with correctness status.

## Phase 2: Triton Performance Iteration

1. Profile current Triton-path baseline latency distribution on 4090 (small and large `seq_len` cases).
2. If baseline is too slow, incrementally move hot portions into Triton kernels (routing/topk or fused math pieces), while preserving exact output semantics.
3. Re-run local benchmark after each optimization; keep only changes that improve speed without correctness regressions.

## Phase 3: CUDA Path Implementation

1. Implement matching entry in `/workspace/flashinfer2026contest/solution/cuda/binding.py` with the identical argument contract as Triton path.
2. Implement `/workspace/flashinfer2026contest/solution/cuda/kernel.cu` for the same FP8 block-scale MoE semantics (routing + grouped GEMMs + accumulation), starting with correctness-first decomposition.
3. Wire launch configuration and pointer handling in binding, then validate local benchmark correctness first.
4. Optimize CUDA kernel launch parameters and memory movement for 4090; re-benchmark and compare against Triton path.

## Validation and Delivery

- Keep `config.toml` switched per run (`language = "triton"` then `"cuda"`) and produce validated `solution.json` for each path.
- Report per-workload status, correctness errors, and speedups from `scripts/run_local.py`.
- Leave both implementations in repo so you can iterate/submit either path quickly.

## Key References

- Track A kernel page: https://bench.flashinfer.ai/kernels/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
- Dataset and local setup: https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
- FP8 MoE API reference: https://docs.flashinfer.ai/api/fused_moe.html
- `trtllm_fp8_block_scale_moe` docs: https://docs.flashinfer.ai/generated/flashinfer.fused_moe.trtllm_fp8_block_scale_moe.html
- BYOK workflow: https://flashinfer-bench.mintlify.app/docs/tutorials/bring_your_own_kernel
