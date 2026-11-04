"""SparseVLM (Zhang et al., ICML 2025): text-guided, training-free, progressive visual-token sparsification.

Ported from llava/model/language_model/{score.py, modelling_sparse_llama.py} (commit a9e7142, Apache-2.0, v1.0 =
the ICML paper):
- Text raters (paper §3.2): on the decoder input embeddings, m = softmax over text of (visual x text), averaged over
  visual tokens; the raters are the text tokens after the image with m above its mean.
- Stages at decoder layers 2, 6, 15 (`pruning_loc`): the layer's attention, averaged over heads and over the raters,
  ranks the surviving visual tokens (`attn_postprocess_topk`); the top tokens survive from the next layer on.
- Kept counts follow `sparse_token_list_64` = (66, 30, 17) of 576 (the 64-token setting, ~11% kept).

Deviations:
- Budget: the three counts keep their official ratios and are scaled so the mean number of visual tokens per decoder
  layer equals FastV's at the same budget (plan.md §C3, the rule PDrop already uses). Official counts are absolute
  for LLaVA's 576 tokens; our images have 64-1,300.
- No token recycling, original position ids kept: v1.0 merges 30% of the pruned tokens into clusters and renumbers
  the remaining positions contiguously, which is undefined for M-RoPE (3-D position ids). The authors' v2.0
  (`USE_VERSION=2_0`) makes the same two choices (`s_flag = False`, "Keep Position ID").
"""

from __future__ import annotations

import numpy as np
import torch

PRUNING_LAYERS = (2, 6, 15)  # pruning_loc: scores read at these layers; tokens leave from the next layer on
RETAIN_64 = (66, 30, 17)  # sparse_token_list_64 (v1.0): visual tokens kept after each stage, of 576


def sparsevlm_schedule(budget: float, n_layers: int, first: int = 2) -> list[tuple[int, int, float]]:
    """[(score_layer, mask_from, keep_fraction)] with the official stage ratios, token-matched to FastV at `budget`:
    mean visual tokens per layer = (first + (L - first) x budget) / L."""
    target = first + (n_layers - first) * budget  # in units of N tokens x layers
    bounds = [layer + 1 for layer in PRUNING_LAYERS]
    spans = [bounds[1] - bounds[0], bounds[2] - bounds[1], n_layers - bounds[2]]
    rel = [c / RETAIN_64[0] for c in RETAIN_64]
    a = (target - bounds[0]) / sum(s * r for s, r in zip(spans, rel))
    if a <= 0:
        raise ValueError(f"budget {budget} is below SparseVLM's three full layers on a {n_layers}-layer decoder")
    return [(layer, b, min(1.0, a * r)) for layer, b, r in zip(PRUNING_LAYERS, bounds, rel)]


def select_raters(visual: torch.Tensor, text: torch.Tensor) -> np.ndarray:
    """Indices (into `text`) of the text raters. visual [N, d], text [T, d]: decoder input embeddings."""
    m_v_t = visual.float() @ text.float().T  # [N, T]
    m_v_t = m_v_t.softmax(dim=1).mean(dim=0)  # [T]   (official: softmax(2).mean(1) on [1, N, T])
    return torch.where(m_v_t > m_v_t.mean())[0].cpu().numpy()


def stage_keep(relation: np.ndarray, survivors: np.ndarray, count: int) -> np.ndarray:
    """attn_postprocess_topk on the surviving tokens: top `min(count, survivors - 1)` by rater attention."""
    alive = np.flatnonzero(survivors)
    count = max(1, min(count, alive.size - 1))
    order = alive[np.argsort(-relation[alive], kind="stable")]
    return np.sort(order[:count]).astype(np.int64)
