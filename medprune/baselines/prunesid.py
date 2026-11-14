"""PruneSID (Fang et al., arXiv 2603.09480): Principal Semantic Components grouping + intra-group NMS.

Port of `Qwen2VisionTransformerPretrainedModel_prunesid.forward` (prunesid/prunesid_qwen/modeling_qwen2_vl.py,
commit 4ef17d9, Apache-2.0), the authors' Qwen2-VL plug-in: it selects on the merger output, i.e. the visual
embeddings the LLM receives, and prunes before the LLM (original position ids kept). The selection below is the
official code line for line, operating on those embeddings.

Deviations (both for a matched budget across methods):
- The kept count is passed directly (k = 10% of N) instead of `need_token_num` x N / 576 with a floor of 16 tokens;
  the floor would give PruneSID 25% of RadImageNet's 64 tokens.
- `torch.pca_lowrank` is randomized; the official code does not seed it. Here it runs on CPU in float32 under a
  per-question seed, so a run is reproducible.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from . import keep_pool


def pca_group(features: torch.Tensor, min_components: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    standard_features = torch.sigmoid(features.to(torch.float32)).permute(1, 0)
    U, S, V = torch.pca_lowrank(standard_features, q=min_components)
    V = torch.abs(V)
    belong_components = torch.argmax(V, dim=1)
    return V, belong_components


def nms(similarity_matrix: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    keep = []
    while scores.sum() > 0:
        max_idx = scores.argmax(axis=0)
        scores[max_idx] = 0
        keep.append(max_idx)
        condition = similarity_matrix[max_idx] > threshold
        scores[condition] = 0
    return keep


def prunesid_indices(hidden_states: torch.Tensor, need_token_num: int) -> torch.Tensor:
    """The official selection on visual embeddings [N, C]; returns kept indices (official order)."""
    projector_lengths, belong_components = pca_group(hidden_states, min_components=max(int(need_token_num / 4), 4))
    projector_scores = projector_lengths.clone()

    projector_mask = belong_components.unsqueeze(1).repeat(1, projector_lengths.shape[1])
    index_map = torch.arange(projector_lengths.shape[1], device=projector_lengths.device).unsqueeze(0) \
        .repeat(projector_lengths.shape[0], 1)
    weights_mask = torch.where(projector_mask != index_map)
    projector_lengths[weights_mask] = 0
    projector_scores[weights_mask] = 0

    normalized_states = F.normalize(hidden_states, p=2, dim=-1)
    group_similarity = torch.bmm(normalized_states.unsqueeze(0), normalized_states.T.unsqueeze(0))[0]
    sim_mean = group_similarity.triu(diagonal=0).mean()
    group_similarity = group_similarity.to(torch.float32).cpu().numpy()
    group_idxs = []

    ratio = max(need_token_num / ((hidden_states.shape[0]) / 18), 1)
    given_scores = torch.arange(projector_lengths.shape[0], 0, -1, device=projector_lengths.device,
                                dtype=projector_lengths.dtype)
    for g in range(projector_lengths.shape[1]):
        group_indices = torch.where(belong_components == g)[0].cpu().numpy()
        if group_indices.shape[0] == 0:
            group_idxs.append(np.array([]))
            continue
        g_similarity = group_similarity[group_indices, :][:, group_indices]
        g_scores = projector_lengths[:, g][group_indices].cpu().numpy()
        keep_indices = nms(g_similarity, g_scores, float(ratio * sim_mean))
        keep_indices = group_indices[keep_indices]
        projector_scores[keep_indices, g] = given_scores[:keep_indices.shape[0]]
        group_idxs.append(keep_indices)

    keep_nms_counts = torch.tensor([group_idxs[i].shape[0] for i in range(len(group_idxs))],
                                   device=projector_lengths.device)
    group_counts = (projector_mask == index_map).sum(dim=0)

    group_lower_bound = torch.ones(group_counts.shape[0], device=group_counts.device)
    group_lower_bound = torch.min(torch.cat([group_lower_bound.unsqueeze(0), group_counts.unsqueeze(0)], dim=0),
                                  dim=0)[0]

    group_upper_bound = torch.ones(group_counts.shape[0], device=group_counts.device) * 5 * math.ceil(need_token_num / 64)
    group_upper_bound = torch.min(torch.cat([group_upper_bound.unsqueeze(0), group_counts.unsqueeze(0)], dim=0),
                                  dim=0)[0]
    group_upper_bound = torch.min(torch.cat([group_upper_bound.unsqueeze(0), keep_nms_counts.unsqueeze(0)], dim=0),
                                  dim=0)[0]
    while group_upper_bound.sum() < need_token_num:
        group_upper_bound = group_upper_bound + 1
        group_upper_bound = torch.min(torch.cat([group_upper_bound.unsqueeze(0), group_counts.unsqueeze(0)], dim=0),
                                      dim=0)[0]

    other_token_nums = max(0, need_token_num - group_lower_bound.sum())
    norm_group_counts = keep_nms_counts / keep_nms_counts.sum()
    cumulative_sum = torch.cumsum(norm_group_counts, dim=0)
    other_token_d = (cumulative_sum * other_token_nums).round().int()
    other_token_d = other_token_d - torch.cat([torch.zeros(1, device=other_token_d.device), other_token_d[:-1]])
    group_token_d = other_token_d + group_lower_bound

    group_token_d = torch.min(torch.cat([group_token_d.unsqueeze(0), group_upper_bound.unsqueeze(0)], dim=0), dim=0)[0]
    group_mean_sort_index = torch.argsort(keep_nms_counts, descending=True, dim=0)
    filling_group = 0

    while group_token_d.sum() < other_token_nums + group_lower_bound.sum():
        filling_num = min(group_upper_bound[group_mean_sort_index[filling_group]]
                          - group_token_d[group_mean_sort_index[filling_group]],
                          other_token_nums + group_lower_bound.sum() - group_token_d.sum())
        group_token_d[group_mean_sort_index[filling_group]] += filling_num
        filling_group += 1
    projector_sort_index = torch.argsort(projector_scores, descending=True, dim=0)
    important_indices = []
    for g in range(len(group_token_d)):
        important_indices.append(projector_sort_index[:, g][:int(group_token_d[g])])
    important_indices = torch.cat(important_indices, dim=0)[:need_token_num]
    if important_indices.shape[0] > 150:
        important_indices = important_indices.sort()[0]
    return important_indices


def select_prunesid(feats: np.ndarray, k: int, seed: int, allowed: np.ndarray | None = None) -> np.ndarray:
    """k kept visual tokens (sorted indices) from LLM-input visual embeddings [N, C].

    With `allowed` (gate_prunesid), the official selection runs on the allowed tokens only (the 16-token floor is the
    image's, so it is tested on N); a pool of at most k tokens, or one too small for the PCA, is kept whole and
    topped up in raster order, PruneSID's own shortfall rule."""
    n = feats.shape[0]
    if n <= 16 or k >= n:  # official: images of <= 16 tokens are not pruned
        return np.arange(n, dtype=np.int64)
    pool = np.arange(n) if allowed is None else np.flatnonzero(allowed)
    if pool.size <= max(k, int(k / 4), 4):
        return keep_pool(pool[:k], k, np.arange(n))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        idx = prunesid_indices(torch.as_tensor(feats[pool], dtype=torch.float32), k).cpu().numpy()
    chosen = np.unique(pool[idx.astype(np.int64)])
    if chosen.size < k:  # the group allocation can fall short when NMS leaves few tokens; top up in raster order
        rest = np.concatenate([pool, np.arange(n)])  # pool first: a gated top-up stays in the foreground if it can
        rest = rest[~np.isin(rest, chosen)]
        _, first = np.unique(rest, return_index=True)
        rest = rest[np.sort(first)]
        chosen = np.concatenate([chosen, rest[: k - chosen.size]])
    return np.sort(chosen[:k])
