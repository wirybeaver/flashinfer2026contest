"""
Track A (fused_moe) Triton-path implementation.

For this starter kit, the packaged entrypoint should expose a Python
`kernel(...)` function matching the Track A definition.
"""

from __future__ import annotations

import torch


def _ensure_shape(tensor: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} shape mismatch, expected {expected}, got {tuple(tensor.shape)}")


def _block_dequant_matrix(
    matrix_fp8: torch.Tensor,
    scales: torch.Tensor,
    num_row_blocks: int,
    num_col_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Dequantize a [R, C] block-scaled matrix without materializing repeated scales."""
    return (
        matrix_fp8.to(torch.float32)
        .view(num_row_blocks, block_size, num_col_blocks, block_size)
        .mul(scales.to(torch.float32).view(num_row_blocks, 1, num_col_blocks, 1))
        .reshape(num_row_blocks * block_size, num_col_blocks * block_size)
    )


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
    output: torch.Tensor,
) -> None:
    """
    Track A entrypoint:
      moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
    """
    # Fixed geometry/constants from the Track A definition.
    hidden_size = 7168
    intermediate_size = 2048
    num_experts = 256
    top_k = 8
    n_group = 8
    topk_group = 4
    block_size = 128
    local_num_experts = 32

    seq_len = int(routing_logits.shape[0])
    # Shape reference (kept as comments for readability/documentation):
    # _ensure_shape(routing_logits, (seq_len, num_experts), "routing_logits")
    # _ensure_shape(routing_bias, (num_experts,), "routing_bias")
    # _ensure_shape(hidden_states, (seq_len, hidden_size), "hidden_states")
    # _ensure_shape(hidden_states_scale, (hidden_size // block_size, seq_len), "hidden_states_scale")
    # _ensure_shape(gemm1_weights, (local_num_experts, 2 * intermediate_size, hidden_size), "gemm1_weights")
    # _ensure_shape(
    #     gemm1_weights_scale,
    #     (local_num_experts, (2 * intermediate_size) // block_size, hidden_size // block_size),
    #     "gemm1_weights_scale",
    # )
    # _ensure_shape(gemm2_weights, (local_num_experts, hidden_size, intermediate_size), "gemm2_weights")
    # _ensure_shape(
    #     gemm2_weights_scale,
    #     (local_num_experts, hidden_size // block_size, intermediate_size // block_size),
    #     "gemm2_weights_scale",
    # )

    num_hidden_blocks = hidden_size // block_size  # 56
    num_intermediate_blocks = intermediate_size // block_size  # 16
    num_gemm1_out_blocks = (2 * intermediate_size) // block_size  # 32
    group_size = num_experts // n_group

    a = (
        hidden_states.to(torch.float32)
        .view(seq_len, num_hidden_blocks, block_size)
        .mul(hidden_states_scale.to(torch.float32).permute(1, 0).unsqueeze(-1))
        .reshape(seq_len, hidden_size)
    )

    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).view(1, num_experts)
    s = torch.sigmoid(logits)
    s_with_bias = s + bias

    s_grouped = s_with_bias.view(seq_len, n_group, group_size)
    top2_vals, _ = torch.topk(s_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)
    _, group_idx = torch.topk(group_scores, k=topk_group, dim=1, largest=True, sorted=False)

    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(seq_len, n_group, group_size).reshape(seq_len, num_experts)
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, torch.finfo(torch.float32).min)
    _, topk_idx = torch.topk(scores_pruned, k=top_k, dim=1, largest=True, sorted=False)

    m = torch.zeros_like(s)
    m.scatter_(1, topk_idx, 1.0)
    weights = s * m
    weights = (weights / (weights.sum(dim=1, keepdim=True) + 1e-20)) * float(routed_scaling_factor)

    accum = torch.zeros((seq_len, hidden_size), dtype=torch.float32, device=hidden_states.device)
    local_start = int(local_expert_offset)

    for local_expert_id in range(local_num_experts):
        global_expert_id = local_start + local_expert_id
        if global_expert_id < 0 or global_expert_id >= num_experts:
            continue

        token_mask = (topk_idx == global_expert_id).any(dim=1)
        if not token_mask.any():
            continue

        token_idx = torch.nonzero(token_mask, as_tuple=False).squeeze(1)
        a_expert = a.index_select(0, token_idx)

        w13 = _block_dequant_matrix(
            gemm1_weights[local_expert_id],
            gemm1_weights_scale[local_expert_id],
            num_gemm1_out_blocks,
            num_hidden_blocks,
            block_size,
        )
        w2 = _block_dequant_matrix(
            gemm2_weights[local_expert_id],
            gemm2_weights_scale[local_expert_id],
            num_hidden_blocks,
            num_intermediate_blocks,
            block_size,
        )

        g1 = a_expert.matmul(w13.t())
        x1 = g1[:, :intermediate_size]
        x2 = g1[:, intermediate_size:]
        c = torch.nn.functional.silu(x2) * x1
        o = c.matmul(w2.t())

        w_tok = weights.index_select(0, token_idx)[:, global_expert_id]
        accum.index_add_(0, token_idx, o * w_tok.unsqueeze(1))

    output.copy_(accum.to(torch.bfloat16))
