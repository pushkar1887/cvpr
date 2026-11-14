"""VisPruner (Zhang et al., ICCV 2025): important tokens by visual attention, then diverse tokens by removing
duplicates.

Port of `LlavaMetaForCausalLM.encode_images` (llava/model/llava_arch.py, commit aefa01a, Apache-2.0): keep the
T x r tokens with the highest visual attention (r = `important_ratio`, official default 0.5), then shrink the rest by
iterative bipartite matching (8 most redundant tokens per round) until T x (1 - r) remain. Prunes before the LLM.

Adaptation (Lingshu-7B / Qwen3-VL-8B have no [CLS] token): the official score is the CLIP [CLS] token's attention
to each patch at the feature layer (`attentions[select_layer][:, :, 0, 1:]`, head mean). Here the score is the
attention each patch receives at the vision tower's last block — the block whose output the merger turns into LLM
tokens — averaged over heads and over all query patches, then averaged over the 2 x 2 patches of each merged token.
The duplicate-removal features are that block's output, concatenated per merged token (the pre-projector features,
as in the official code). Everything after the score is the official code, with one guard: the official loop
assumes an even number of residual tokens (LLaVA: 576 - 32 = 544, minus 8 per round) and raises an IndexError for
odd counts or fewer than 16 residual tokens; here the 'a' side of the bipartite split is indexed by its actual
length, ceil(R / 2), and at most ceil(R / 2) tokens leave per round. Where the official code runs, the result is
identical (tests/test_baselines.py).
"""

from __future__ import annotations

import numpy as np
import torch

from . import keep_pool

IMPORTANT_RATIO = 0.5  # official default (llava/eval/model_vqa*.py --important_ratio)


def vispruner_indices(image_features: torch.Tensor, image_attentions: torch.Tensor, visual_token_num: int,
                      important_ratio: float = IMPORTANT_RATIO) -> torch.Tensor:
    """Official selection for one image. image_features [1, N, C]; image_attentions [1, H, N] (or [1, 1, N]).
    Returns the boolean index mask [1, N]."""
    B, N, C = image_features.shape
    device = image_features.device

    important_token_num = int(visual_token_num * important_ratio)  # T_imp = T * r
    diverse_token_num = visual_token_num - important_token_num  # T_div = T * (1 - r)

    image_attentions = image_attentions.mean(dim=1)  # (B, N)
    token_indices = image_attentions.argsort(dim=-1, descending=True)  # (B, N)
    important_indices = token_indices[:, :important_token_num]  # (B, T_imp)
    residual_indices = token_indices[:, important_token_num:]  # (B, N - T_imp)

    image_normalized = image_features / image_features.norm(dim=-1, keepdim=True)  # (B, N, C)
    while diverse_token_num > 0:
        R = residual_indices.shape[1]
        r = min(8, R - diverse_token_num, (R + 1) // 2)  # (R + 1) // 2: guard, see the module docstring
        if r <= 0:
            break

        residual_tokens = image_normalized[torch.arange(B).unsqueeze(-1).expand(-1, R), residual_indices]  # (B, R, C)
        a, b = residual_tokens[..., ::2, :], residual_tokens[..., 1::2, :]  # (B, R // 2, C)
        scores = a @ b.transpose(-1, -2)  # (B, R // 2, R // 2)
        scores = scores.max(dim=-1).values  # (B, R // 2)

        distinct_indices = scores.argsort(dim=-1, descending=True)[:, r:]  # (B, ceil(R / 2) - r)
        residual_indices = torch.cat([
            residual_indices[..., ::2][torch.arange(B).unsqueeze(-1).expand(-1, distinct_indices.shape[1]),
                                       distinct_indices],
            residual_indices[..., 1::2]
        ], dim=-1)  # (B, R - r)

    if diverse_token_num > 0:
        selected_indices = torch.cat([important_indices, residual_indices], dim=-1)  # (B, T)
    else:
        selected_indices = important_indices  # (B, T)
    index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
    index_masks.scatter_(1, selected_indices, True)
    return index_masks


def select_vispruner(importance: np.ndarray, feats: np.ndarray, k: int,
                     important_ratio: float = IMPORTANT_RATIO, allowed: np.ndarray | None = None) -> np.ndarray:
    """k kept visual tokens (sorted indices) from per-token visual attention [N] and features [N, C].

    With `allowed` (gate_vispruner), the official selection runs on the allowed tokens only; a pool of at most k
    tokens is kept whole and topped up by visual attention."""
    n = importance.shape[0]
    if k >= n:
        return np.arange(n, dtype=np.int64)
    pool = np.arange(n) if allowed is None else np.flatnonzero(allowed)
    if pool.size <= k:
        return keep_pool(pool, k, np.argsort(-np.asarray(importance, dtype=np.float64), kind="stable"))
    mask = vispruner_indices(torch.as_tensor(feats[pool], dtype=torch.float32)[None],
                             torch.as_tensor(importance[pool], dtype=torch.float32)[None, None], k, important_ratio)
    return np.sort(pool[np.flatnonzero(mask[0].numpy())]).astype(np.int64)
