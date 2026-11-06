"""TransPrune (Li et al., CVPR 2026): token transition variation (TTV) plus instruction-guided attention (IGA),
progressive pruning inside the LLM.

Port of llava/model/modeling_llama_transprune.py (github.com/liaolea/TransPrune, commit c9e8284, MIT):
- TTV of a token at one decoder layer (`compute_transf_weight`), for the attention and the FFN sub-layer each:
  (||after|| / ||before||) x softmax over the sequence of (1 - cos(before, after)), where before = the sub-layer's
  normalized input and after = its output; the layer's TTV is the sum of the two. TTV is accumulated over the
  accumulating layers seen so far and normalized to sum 1 over the surviving visual tokens.
- IGA at a pruning layer: the attention of every prompt token after the image (with RoPE, causal), averaged over
  heads and queries, read from that layer's projections of its input, normalized to sum 1 over the survivors.
- score = TTV + IGA; the top tokens survive from that layer on. Official LLaVA-1.5 schedule: accumulating layers
  7-12 (1-based), pruning at layers 7, 9, 12 (the stages act before 0-based layers 7, 9 and 12), keeping 87.5%,
  62.5% and 12.5% of the visual tokens.

Adaptations: masking replaces removal; the softmax inside TTV runs over the live sequence (masked visual tokens are
excluded, as removed tokens are in the official code), so kept tokens get the official weights. Layers are the
official absolute ones on both models (as for SparseVLM's port). Budget: the last stage keeps the run's budget
instead of 12.5%; the earlier stages keep the official 87.5% and 62.5%. TransPrune cannot be token-matched to FastV
at 10% (its first seven layers are unpruned); FLOPs are reported (medprune.latency).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

PRUNING_LAYERS = (7, 9, 12)  # 0-based index of the first layer each stage's dropped tokens are masked at
ACCUMULATING_LAYERS = tuple(range(6, 12))  # 0-based layers whose TTV is accumulated (official 1-based 7-12)
RATIOS = (0.875, 0.625)  # official image_token_ratio_list for the first two stages; the last one = budget


def transprune_schedule(budget: float) -> list[tuple[int, float]]:
    """[(pruning layer, keep fraction of N)]."""
    return list(zip(PRUNING_LAYERS, (*RATIOS, budget)))


def compute_transf_weight(before_hidden_states, after_hidden_states):
    """Official TTV of one sub-layer over a whole sequence: [B, T] (batch, tokens)."""
    before_norm = torch.norm(before_hidden_states, dim=-1)
    after_norm = torch.norm(after_hidden_states, dim=-1)
    length_ratio = after_norm / before_norm

    cosine_similarities = 1 - F.cosine_similarity(before_hidden_states, after_hidden_states, dim=-1)
    exp_cosine = torch.exp(cosine_similarities)
    necc = exp_cosine / exp_cosine.sum()
    transformation_weights = length_ratio * necc

    return transformation_weights


def transition_weight(before: torch.Tensor, after: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    """compute_transf_weight over the live tokens only. before/after [T, d]; live [T] bool. Returns [T] float32
    (zero where not live)."""
    w = torch.zeros(before.shape[0], dtype=torch.float32, device=before.device)
    w[live] = compute_transf_weight(before[live].float()[None], after[live].float()[None])[0]
    return w


def stage_keep(iga: np.ndarray, ttv: np.ndarray, survivors: np.ndarray, count: int) -> np.ndarray:
    """Top `count` survivors by normalized TTV + normalized IGA (official, `a = 0.5` left commented)."""
    alive = np.flatnonzero(survivors)
    count = max(1, min(count, alive.size))
    t = np.asarray(ttv, dtype=np.float64)[alive]
    a = np.asarray(iga, dtype=np.float64)[alive]
    score = t / (t.sum() + 1e-8) + a / (a.sum() + 1e-8)
    order = alive[np.argsort(-score, kind="stable")]
    return np.sort(order[:count]).astype(np.int64)
