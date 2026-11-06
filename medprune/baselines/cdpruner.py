"""CDPruner (Zhang et al., NeurIPS 2025): instruction-conditioned diversity by MAP inference of a conditional DPP.
Prunes before the LLM and never reads LLM attention.

Port of `LlavaMetaForCausalLM.encode_images` (llava/model/llava_arch.py, commit 9541616, Apache-2.0): kernel
L = diag(r) S diag(r), S = cosine similarity of the projected visual tokens, r = min-max normalized instruction
relevance; fast greedy MAP inference (Chen et al., NeurIPS 2018) picks T tokens.

Relevance for models without a CLIP text tower (paper §3.3, used for Qwen2.5-VL; the released code covers LLaVA
only): visual embeddings = the multimodal projector's outputs, text embedding = the mean of the LLM's input
embeddings of the instruction tokens, r_i = cos(H_v^i, mean H_q) (Eq. 5), then min-max normalization (Eq. 6). The
LLaVA code negates the CLIP image-text similarity before normalizing; Eq. 5 has no negation, and the paper gives no
negation for the projector-embedding variant, so `cdpruner` applies none. `cdpruner_neg` (Nov 7 sensitivity run) uses
the released code's sign, r_i = -cos(H_v^i, mean H_q): on Qwen3-VL RadImageNet anatomy the unnegated relevance keeps
background tokens at 2.5x chance. Instruction tokens = the user turn's text
after the image (question and answer-format line), without the chat template.
With a foreground gate (gate_cdpruner), the DPP runs on the gate's foreground tokens only.
"""

from __future__ import annotations

import numpy as np
import torch


def cdpruner_indices(image_features: torch.Tensor, relevance: torch.Tensor, visual_token_num: int) -> torch.Tensor:
    """Official kernel and greedy MAP (batch 1). image_features [1, N, D] (projected); relevance [1, N] before
    min-max normalization. Returns the boolean index mask [1, N]."""
    B, N, C = image_features.shape
    device = image_features.device

    # [CDPruner] Calculate cosine similarity
    image_normalized = image_features / image_features.norm(dim=-1, keepdim=True)  # (B, N, D)
    image_normalized = image_normalized.float()  # (B, N, D)
    similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2))  # (B, N, N)

    relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min())  # (B, N)

    # [CDPruner] Construct kernel matrix
    kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)  # (B, N, N)

    # [CDPruner] Fast MAP inference of conditional DPP
    cis = torch.zeros((visual_token_num, B, N), device=device)  # (T, B, N)
    di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone()  # (B, N)
    select_idx = torch.empty((visual_token_num, B), dtype=torch.long, device=device)  # (T, B)
    for i in range(visual_token_num):
        j = torch.argmax(di2s, dim=-1)
        select_idx[i] = j

        eis = (kernel[torch.arange(B), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B), j], cis[:i])) \
            / torch.sqrt(di2s[torch.arange(B), j]).unsqueeze(-1)
        cis[i, :, :] = eis
        di2s -= torch.square(eis)
        di2s[torch.arange(B), j] = -float('inf')

    select_idx = torch.sort(select_idx.t()).values  # (B, T)
    index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
    index_masks.scatter_(1, select_idx, True)
    return index_masks


def instruction_relevance(visual: np.ndarray, text: np.ndarray) -> np.ndarray:
    """r_i = cos(H_v^i, mean_j H_q^j) (paper Eq. 5). visual [N, d]; text [M, d] (LLM input embeddings)."""
    v = torch.as_tensor(visual, dtype=torch.float32)
    q = torch.as_tensor(text, dtype=torch.float32).mean(dim=0, keepdim=True)
    v, q = v / v.norm(dim=-1, keepdim=True), q / q.norm(dim=-1, keepdim=True)  # official order: normalize, then dot
    return (v @ q.T)[:, 0].numpy()


def select_cdpruner(visual: np.ndarray, text: np.ndarray, k: int, allowed: np.ndarray | None = None,
                    negate: bool = False) -> np.ndarray:
    """k kept tokens (sorted indices) from projected visual tokens [N, d] and instruction embeddings [M, d].
    `negate`: the released LLaVA code's sign (relevance = -cosine)."""
    n = visual.shape[0]
    if k >= n:
        return np.arange(n, dtype=np.int64)
    pool = np.arange(n) if allowed is None else np.flatnonzero(allowed)
    rel = instruction_relevance(visual, text)
    rel = -rel if negate else rel
    if pool.size <= k:  # gate too tight: keep the pool, top up by relevance outside it
        order = np.argsort(-rel, kind="stable")
        rest = order[~np.isin(order, pool)][: k - pool.size]
        return np.sort(np.concatenate([pool, rest])).astype(np.int64)
    mask = cdpruner_indices(torch.as_tensor(visual[pool], dtype=torch.float32)[None],
                            torch.as_tensor(rel[pool], dtype=torch.float32)[None], k)
    return np.sort(pool[np.flatnonzero(mask[0].numpy())]).astype(np.int64)
