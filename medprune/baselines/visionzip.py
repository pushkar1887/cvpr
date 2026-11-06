"""VisionZip (Yang et al., CVPR 2025): dominant tokens by vision-encoder attention, plus contextual tokens that merge
the rest by key similarity. Prunes before the LLM.

Port of the VisionZip block of `Qwen2_5_VLForConditionalGeneration.forward` (Qwen2_5_VL/qwen2_5vl_visionzip.py,
commit 8f86b55, Apache-2.0), the authors' own Qwen2.5-VL version:
- score: attention each merged token receives at the vision tower's last block (head mean, summed over queries,
  merge-unit mean); key metric: that block's (RoPE'd) keys, merge-unit and head mean;
- dominant: the top `dominant_num` tokens by score; contextual: among the rest (raster order), `contextual_num`
  targets at stride max(1, R // contextual_num); every other non-dominant token joins its most cosine-similar
  target (normalized keys); each target's LLM-input embedding becomes target + mean(joined embeddings), at the
  target's position;
- counts: contextual_num = max(int(0.05 N), 1) and dominant_num = the rest of the budget (the Qwen code sets
  dominant 65% / 45% + contextual 5%; here dominant + contextual = the run's budget).

Adaptations: masking from layer 0 replaces removal (identical, tests/test_attention_hook.py); the merged embedding
is written into the vision tower's output (medprune.models applies `state.merge`). Qwen3-VL (no released code):
the same selection on its last vision block, and the same merge weights applied to its deepstack features, so a
contextual token carries merged features into every layer that receives visual features.
With a foreground gate (gate_visionzip), dominant, target and joined tokens all come from the gate's foreground;
background tokens are dropped unmerged.
"""

from __future__ import annotations

import numpy as np
import torch

CONTEXTUAL_RATIO = 0.05  # contextual_num = max(int(0.05 * N), 1), official Qwen2.5-VL code


def visionzip_plan(attn_logits: torch.Tensor, attn_key: torch.Tensor, dominant_num: int, contextual_num: int):
    """Official selection (batch 1). attn_logits [N]; attn_key [1, N, D].
    Returns (select_mask [N] bool, target_positions [C] (indices into 0..N-1), assign_one_hot [1, M, C],
    merge_positions [M] (indices into 0..N-1 of the merged tokens))."""
    topk_values, topk_indices = torch.topk(attn_logits, dominant_num)

    mask = torch.zeros_like(attn_logits, dtype=torch.bool)
    mask[topk_indices] = True
    contextual_mask = ~mask
    metric_filtered = attn_key[:, contextual_mask]
    metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True)

    ## Contextual Visual Tokens
    step = max(1, metric_normalized.shape[1] // contextual_num)
    target_indices = torch.arange(0, metric_normalized.shape[1], step, device=metric_normalized.device)[:contextual_num]
    target_tokens = metric_normalized[:, target_indices, :]

    tokens_to_merge = metric_normalized[:, ~torch.isin(torch.arange(metric_normalized.shape[1],
                                                                    device=metric_normalized.device), target_indices), :]
    similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
    assign_one_hot = torch.zeros(tokens_to_merge.shape[0], tokens_to_merge.shape[1], contextual_num,
                                 dtype=attn_logits.dtype, device=metric_normalized.device)
    assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)

    select_mask = torch.zeros_like(attn_logits, dtype=torch.bool)
    select_mask[topk_indices] = True
    false_pos = (~select_mask).nonzero(as_tuple=True)[0]
    select_mask[false_pos[target_indices]] = True

    merge_positions = false_pos[~torch.isin(torch.arange(false_pos.shape[0], device=false_pos.device), target_indices)]
    return select_mask, false_pos[target_indices], assign_one_hot, merge_positions


def contextual_tokens(embeds: torch.Tensor, targets: torch.Tensor, assign_one_hot: torch.Tensor,
                      merge_positions: torch.Tensor) -> torch.Tensor:
    """Official merge: target + mean of the tokens assigned to it. embeds [N, d] -> [C, d]."""
    counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)  # [1, C, 1]
    hidden_to_merge = embeds[merge_positions][None]  # [1, M, d]
    aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2).to(embeds.dtype), hidden_to_merge) / counts
    return embeds[targets] + aggregated_hidden[0]


def select_visionzip(attn: np.ndarray, keys: np.ndarray, k: int, allowed: np.ndarray | None = None):
    """k kept tokens and the merge as (rows, weights): embedding[rows] <- weights @ embedding, weights [C, N].

    attn [N], keys [N, D]. With `allowed`, the official selection runs on the allowed tokens only (gate)."""
    n = attn.shape[0]
    if k >= n:
        return np.arange(n, dtype=np.int64), None
    pool = np.arange(n) if allowed is None else np.flatnonzero(allowed)
    if pool.size <= k:  # gate leaves no room to merge: keep the pool, top up by attention outside it
        order = np.argsort(-attn, kind="stable")
        rest = order[~np.isin(order, pool)][: k - pool.size]
        return np.sort(np.concatenate([pool, rest])).astype(np.int64), None
    m = pool.size
    contextual_num = min(max(int(CONTEXTUAL_RATIO * m), 1), k - 1) if k > 1 else 0
    dominant_num = k - contextual_num
    a = torch.as_tensor(attn[pool], dtype=torch.float32)
    if contextual_num == 0:
        kept = pool[torch.topk(a, dominant_num).indices.numpy()]
        return np.sort(kept).astype(np.int64), None
    key = torch.as_tensor(keys[pool], dtype=torch.float32)[None]
    select_mask, targets, one_hot, merged = visionzip_plan(a, key, dominant_num, contextual_num)
    kept = pool[np.flatnonzero(select_mask.numpy())]
    # weights: row c = e_target + (1 / count_c) * sum of assigned one-hots, in pool coordinates -> 0..N-1
    counts = one_hot[0].sum(dim=0).clamp(min=1)  # [C]
    weights = np.zeros((contextual_num, n), dtype=np.float64)
    weights[np.arange(contextual_num), pool[targets.numpy()]] = 1.0
    assign = (one_hot[0] / counts).T.numpy()  # [C, M]
    weights[:, pool[merged.numpy()]] += assign
    return np.sort(kept).astype(np.int64), (pool[targets.numpy()].astype(np.int64), weights)
