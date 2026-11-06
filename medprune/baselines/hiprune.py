"""HiPrune (Liu et al., arXiv 2508.00553; AAAI 2026 student abstract, ACL 2026 Findings): anchor, buffer and register
tokens from the vision encoder's hierarchical attention. Prunes before the LLM.

Port of the HiPrune block of `Qwen2_5_VLForConditionalGeneration.forward` (Qwen2_5_VL/qwen2_5_vl_HiPrune.py, commit
8278100, MIT): the attention each merged token receives (head mean, query mean, merge-unit mean, as in the official
vision tower) at the object layer and at the output layer; anchors = the top round(T x alpha / 5) tokens at the object
layer; buffers = their 4-neighbours in the token grid (raster index +-1 and +-width, clamped to [0, N-1], as in the
official code: +-1 can wrap to the neighbouring row); registers = the top remaining tokens at the output layer until
T = round(N x retention) tokens are kept. Official defaults for Qwen2.5-VL: alpha 0.1, object layer 16 of 32.

Adaptations: retention = the run's budget. Qwen3-VL's vision tower (27 blocks, no released HiPrune setting) uses the
same relative depth, object layer 14 (1-based) = round(27 x 16 / 32). Both object layers are full-attention blocks
(Qwen2.5-VL: fullatt_block_indexes 7, 15, 23, 31; Qwen3-VL has no window attention).
"""

from __future__ import annotations

import numpy as np
import torch

ALPHA = 0.1  # HIPRUNE_ALPHA, official default
OBJECT_LAYER = {"lingshu7b": 16, "qwen3vl8b": 14}  # 1-based; official 16 for Qwen2.5-VL (32 blocks)


def object_block(model: str) -> int:
    """0-based index of the object (middle) vision block."""
    return OBJECT_LAYER[model] - 1


def hiprune_indices(shallow_attention: torch.Tensor, deep_attention: torch.Tensor, width: int, retain: float,
                    alpha: float = ALPHA) -> torch.Tensor:
    """Official selection (batch 1). shallow/deep_attention [N]; width = token-grid width. Returns kept indices."""
    n_image_tokens = deep_attention.shape[0]
    deep_attention = deep_attention.clone()

    visual_token_num = round(n_image_tokens * retain)
    shallow_token_num = round((visual_token_num * alpha) / 5)

    shallow_token_indices = torch.topk(shallow_attention, k=shallow_token_num).indices
    shallow_token_indices = torch.cat([shallow_token_indices,
                                       shallow_token_indices - 1,
                                       shallow_token_indices + 1,
                                       shallow_token_indices - width,
                                       shallow_token_indices + width])
    shallow_token_indices = shallow_token_indices.clamp(0, n_image_tokens - 1)
    shallow_token_indices = torch.unique(shallow_token_indices, sorted=False)

    deep_token_num = visual_token_num - shallow_token_indices.shape[0]
    available_index_mask = torch.zeros(n_image_tokens, dtype=torch.bool, device=deep_attention.device)
    available_index_mask.scatter_(0, shallow_token_indices, 1)

    deep_attention -= available_index_mask.int()

    deep_token_indices = torch.topk(deep_attention, k=deep_token_num).indices
    return torch.cat([shallow_token_indices, deep_token_indices])


def select_hiprune(shallow: np.ndarray, deep: np.ndarray, grid_hw: tuple[int, int], k: int,
                   alpha: float = ALPHA) -> np.ndarray:
    """k kept tokens (sorted indices). `retain` is chosen so round(N x retain) == k exactly."""
    n = deep.shape[0]
    if k >= n:
        return np.arange(n, dtype=np.int64)
    kept = hiprune_indices(torch.as_tensor(shallow, dtype=torch.float32), torch.as_tensor(deep, dtype=torch.float32),
                           int(grid_hw[1]), k / n, alpha)
    kept = np.unique(kept.numpy()).astype(np.int64)
    if kept.size != k:
        raise RuntimeError(f"HiPrune kept {kept.size} tokens, budget {k}")
    return kept
