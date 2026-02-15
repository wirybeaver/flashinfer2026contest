"""
CUDA-path Python binding helper for Track A.

This mirrors the Triton baseline behavior and is useful for local debugging.
The benchmark CUDA builder itself uses TVM-FFI symbols from `kernel.cu`.
"""

from __future__ import annotations

import torch
from flashinfer.fused_moe import trtllm_fp8_block_scale_moe


def _ensure_shape(tensor: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} shape mismatch, expected {expected}, got {tuple(tensor.shape)}")


def kernel(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
) -> torch.Tensor:
    hidden_size = 7168
    intermediate_size = 2048
    num_experts = 256
    top_k = 8
    n_group = 8
    topk_group = 4
    block_size = 128
    local_num_experts = 32

    seq_len = int(routing_logits.shape[0])
    _ensure_shape(routing_logits, (seq_len, num_experts), "routing_logits")
    _ensure_shape(routing_bias, (num_experts,), "routing_bias")
    _ensure_shape(hidden_states, (seq_len, hidden_size), "hidden_states")
    _ensure_shape(hidden_states_scale, (hidden_size // block_size, seq_len), "hidden_states_scale")
    _ensure_shape(gemm1_weights, (local_num_experts, 2 * intermediate_size, hidden_size), "gemm1_weights")
    _ensure_shape(
        gemm1_weights_scale,
        (local_num_experts, (2 * intermediate_size) // block_size, hidden_size // block_size),
        "gemm1_weights_scale",
    )
    _ensure_shape(gemm2_weights, (local_num_experts, hidden_size, intermediate_size), "gemm2_weights")
    _ensure_shape(
        gemm2_weights_scale,
        (local_num_experts, hidden_size // block_size, intermediate_size // block_size),
        "gemm2_weights_scale",
    )

    tune_max_num_tokens = 16384 if seq_len > 8192 else 8192

    output = trtllm_fp8_block_scale_moe(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        num_experts=num_experts,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        intermediate_size=intermediate_size,
        local_expert_offset=int(local_expert_offset),
        local_num_experts=local_num_experts,
        routed_scaling_factor=float(routed_scaling_factor),
        routing_method_type=0,
        tune_max_num_tokens=tune_max_num_tokens,
    )
    return output.to(torch.bfloat16)
