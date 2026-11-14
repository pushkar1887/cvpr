"""VScan (Zhang et al., TMLR 2026): complementary global and local scans with token merging in the vision encoder,
then text-guided pruning at a middle LLM layer.

Port of the Qwen2.5-VL code (VScan/qwen/model/qwen2_5_vl_custom.py and qwen2_5_vl_utils.py, Tencent/SelfEvolvingAgent
commit a82450e, MIT):
- Stage 1, before the LLM: of T1 kept tokens, T1 // 2 by `window_selection` on the attention each merged token
  receives at the first full-attention vision block (block 7; local scan, 4 x 4-token windows, at most
  ceil(T1/2 / windows) per window) and the rest by top attention at the last block (global scan), excluding the
  local picks; every other token is merged into its most cosine-similar kept token (`token_merging`, scaling 1:
  the kept token's embedding becomes the mean of itself and the tokens merged into it).
- Stage 2, at LLM layer L/2 (14 of 28, the official Qwen2.5-VL setting): the last prompt token's attention without
  RoPE (the official code comments RoPE out), from that layer's q/k projections, head mean, ranks the survivors;
  the top 1/3 survive (official image_token_ratio_list [0.333]).

Budget: official Qwen2.5-VL values keep 16.7% then 1/3 of that (11.1% of tokens per layer on average). Here T1 keeps
the official 1/3 ratio and is scaled so the mean number of visual tokens per decoder layer equals FastV's at the
same budget (plan.md §C3, as for PDrop, SparseVLM and BTP). Qwen3-VL (no released code): local scan at vision block
6 of 27 (the official block 7 of 32 at the same relative depth; Qwen3-VL has no window attention), stage 2 at layer
18 of 36, and the stage-1 merge weights applied to its deepstack features too.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

STAGE2_RATIO = 0.333  # image_token_ratio_list [0.333]
WINDOW = 4
LOCAL_BLOCK = {"lingshu7b": 7, "qwen3vl8b": 6}  # 0-based vision block of the local scan


def vscan_schedule(budget: float, n_layers: int, first: int = 2, layer: int | None = None) -> tuple[float, int, float]:
    """(stage-1 keep fraction, stage-2 layer, stage-2 keep fraction of N), token-matched to FastV at `budget`.

    `layer` moves the second stage (default L/2, VScan's); the fractions are re-solved so the mean visual tokens per
    layer still equal FastV's (plan.md E38).
    """
    layer = n_layers // 2 if layer is None else layer
    target = (first + (n_layers - first) * budget) / n_layers  # FastV's mean visual tokens per layer, units of N
    a = target * n_layers / (layer + (n_layers - layer) * STAGE2_RATIO)
    return min(1.0, a), layer, min(1.0, a) * STAGE2_RATIO


def token_merging(image_embeds, keep_indices, scaling=1):
    """Official (qwen2_5_vl_utils.py): merge every non-retained token into its most similar retained token."""
    N, D = image_embeds.shape
    T = len(keep_indices)

    keep_index_mask = torch.zeros(N, dtype=torch.bool, device=image_embeds.device)
    keep_index_mask[keep_indices] = True

    retained_tokens = image_embeds[keep_index_mask, :]  # [T, D]
    non_retained_tokens = image_embeds[~keep_index_mask, :]  # [N - T, D]

    cosine_sim = F.cosine_similarity(non_retained_tokens.unsqueeze(1), retained_tokens.unsqueeze(0), dim=2)
    nearest_token_indices = cosine_sim.argmax(dim=1)  # [N - T]

    merged_features = torch.zeros_like(retained_tokens)  # [T, D]
    merged_features += retained_tokens * scaling

    expanded_indices = nearest_token_indices  # [N - T]
    merged_features.scatter_add_(0, expanded_indices.unsqueeze(-1).expand(-1, D), non_retained_tokens)

    merge_count = torch.zeros(T, device=image_embeds.device, dtype=torch.int)  # [T]
    merge_count.scatter_add_(0, expanded_indices, torch.ones_like(expanded_indices, dtype=merge_count.dtype))
    merged_features /= (scaling + merge_count.unsqueeze(1))

    return merged_features


def merge_weights(image_embeds: torch.Tensor, keep_indices: torch.Tensor, scaling: float = 1.0) -> np.ndarray:
    """token_merging as a linear map: merged = W @ image_embeds, W [T, N] (rows in sorted keep order)."""
    N = image_embeds.shape[0]
    keep_index_mask = torch.zeros(N, dtype=torch.bool)
    keep_index_mask[keep_indices] = True
    kept, rest = keep_index_mask.nonzero().squeeze(1), (~keep_index_mask).nonzero().squeeze(1)
    x = image_embeds.float()
    nearest = F.cosine_similarity(x[rest].unsqueeze(1), x[kept].unsqueeze(0), dim=2).argmax(dim=1)
    W = np.zeros((kept.numel(), N), dtype=np.float64)
    W[np.arange(kept.numel()), kept.numpy()] = scaling
    np.add.at(W, (nearest.numpy(), rest.numpy()), 1.0)
    return W / W.sum(axis=1, keepdims=True)


def window_selection(attn_weights, num_keep_tokens, token_h: int, token_w: int, window_size=WINDOW):
    """Official (qwen2_5_vl_utils.py), with the merged-grid size passed directly instead of image_grid_thw // 2."""
    assert token_h * token_w == attn_weights.shape[0], "The number of tokens in the window is not equal to num_keep_tokens"
    token_h, token_w = torch.tensor(token_h), torch.tensor(token_w)

    num_windows_h = (token_h / window_size).floor().int()
    num_windows_w = (token_w / window_size).floor().int()
    num_windows = num_windows_h * num_windows_w

    sorted_indices = torch.argsort(attn_weights, dim=0, descending=True)
    window_counter = torch.zeros(int(token_h), int(token_w), device=attn_weights.device, dtype=torch.int)
    total_counter = 0

    # Calculate the limit of the number of tokens to keep in each window
    limit = (num_keep_tokens / num_windows).ceil().int()

    keep_indices = torch.zeros(num_keep_tokens, device=attn_weights.device, dtype=torch.int)
    for index in sorted_indices:
        x = (index // token_w) // window_size
        y = (index % token_w) // window_size
        if x == num_windows_h:
            x = num_windows_h - 1
        if y == num_windows_w:
            y = num_windows_w - 1
        x = x.int()
        y = y.int()
        if window_counter[x, y] < limit:
            window_counter[x, y] += 1
            keep_indices[total_counter] = index
            total_counter += 1
        if total_counter >= num_keep_tokens:
            break

    assert total_counter == num_keep_tokens, "The number of tokens to keep is not equal to num_keep_tokens"
    return keep_indices


def gated_window_selection(attn: np.ndarray, num_keep: int, token_h: int, token_w: int, allowed: np.ndarray,
                           window_size: int = WINDOW) -> np.ndarray:
    """window_selection over the allowed tokens only (gate_vscan): the same windows and per-window limit, candidates
    in descending attention; if the limit leaves fewer than `num_keep` allowed tokens, the best remaining allowed
    tokens fill the rest regardless of window. Equals the official function when every token is allowed."""
    nh, nw = token_h // window_size, token_w // window_size
    limit = -(-num_keep // (nh * nw))
    counter = np.zeros((nh, nw), dtype=int)
    order = np.argsort(-np.asarray(attn, dtype=np.float64), kind="stable")
    order = order[np.asarray(allowed, dtype=bool)[order]]
    keep = []
    for index in order:
        x = min((index // token_w) // window_size, nh - 1)
        y = min((index % token_w) // window_size, nw - 1)
        if counter[x, y] < limit:
            counter[x, y] += 1
            keep.append(int(index))
        if len(keep) >= num_keep:
            break
    if len(keep) < num_keep:
        keep += [int(i) for i in order if i not in set(keep)][: num_keep - len(keep)]
    return np.asarray(keep, dtype=np.int64)


def stage1(local: np.ndarray, global_: np.ndarray, embeds: np.ndarray, grid_hw: tuple[int, int], k1: int,
           allowed: np.ndarray | None = None):
    """Stage-1 kept tokens (sorted) and the merge (rows, weights). local/global_ [N] attention; embeds [N, d].

    With `allowed` (gate_vscan), local and global scans pick allowed tokens only and only dropped allowed tokens are
    merged (into their most similar kept allowed token; `remedy.merge_plan`); dropped background tokens leave unmerged,
    as in gate_visionzip. A pool of at most k1 tokens is kept whole, topped up by global attention, unmerged."""
    n = local.shape[0]
    if k1 >= n:
        return np.arange(n, dtype=np.int64), None
    if allowed is not None:
        return _gated_stage1(local, global_, embeds, grid_hw, k1, np.asarray(allowed, dtype=bool))
    h, w = int(grid_hw[0]), int(grid_hw[1])
    if h < WINDOW or w < WINDOW:  # no complete 4 x 4 window (the official code divides by zero): one window
        keep_local = torch.topk(torch.as_tensor(local, dtype=torch.float32), k1 // 2).indices
    else:
        keep_local = window_selection(torch.as_tensor(local, dtype=torch.float32), k1 // 2, h, w).long()
    g = torch.as_tensor(global_, dtype=torch.float32).clone()
    g[keep_local] = 0  # avoid repetition (official)
    keep_global = torch.topk(g, k1 - k1 // 2).indices
    keep = torch.sort(torch.cat((keep_local, keep_global), dim=0), dim=0)[0]
    if torch.unique(keep).numel() != k1:
        raise RuntimeError("VScan stage 1 selected a token twice")
    weights = merge_weights(torch.as_tensor(embeds), keep)
    return keep.numpy().astype(np.int64), (keep.numpy().astype(np.int64), weights)


def _gated_stage1(local, global_, embeds, grid_hw, k1, allowed):
    from ..remedy import merge_plan
    from . import keep_pool

    pool = np.flatnonzero(allowed)
    if pool.size <= k1:
        return keep_pool(pool, k1, np.argsort(-np.asarray(global_, dtype=np.float64), kind="stable")), None
    h, w = int(grid_hw[0]), int(grid_hw[1])
    if h < WINDOW or w < WINDOW:
        loc = np.where(allowed, np.asarray(local, dtype=np.float64), -np.inf)
        keep_local = np.argsort(-loc, kind="stable")[: k1 // 2]
    else:
        keep_local = gated_window_selection(local, k1 // 2, h, w, allowed)
    g = np.where(allowed, np.asarray(global_, dtype=np.float64), -np.inf)
    g[keep_local] = -np.inf  # avoid repetition (official: set to 0; attention is positive, so the order is the same)
    keep_global = np.argsort(-g, kind="stable")[: k1 - k1 // 2]
    keep = np.sort(np.concatenate([keep_local, keep_global])).astype(np.int64)
    if np.unique(keep).size != k1 or not allowed[keep].all():
        raise RuntimeError("gated VScan stage 1 selected a token twice or outside the gate")
    rows, weights = merge_plan(np.asarray(embeds, dtype=np.float64), keep, allowed)
    return keep, (rows, weights)


def stage2_keep(scores: np.ndarray, survivors: np.ndarray, count: int) -> np.ndarray:
    """Top `count` survivors by the stage-2 (RoPE-free last-token) attention."""
    alive = np.flatnonzero(survivors)
    count = max(1, min(count, alive.size))
    order = alive[np.argsort(-np.asarray(scores, dtype=np.float64)[alive], kind="stable")]
    return np.sort(order[:count]).astype(np.int64)
