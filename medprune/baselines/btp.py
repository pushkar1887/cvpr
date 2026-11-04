"""BTP — Balanced Token Pruning (Li et al., NeurIPS 2025): multi-stage pruning that balances attention (the
current layer's needs) against diversity (later layers' needs), with stage layers fixed on a calibration set.

Independent implementation (the official repository states no license) of the Qwen2.5-VL version,
qwen-2.5-vl/modeling_qwen2_5_vl.py `Qwen2_5_VLModel.forward` (commit 9682db0); tests/test_baselines.py checks the
kept indices against that code, run from a local clone. Stages, with the last prompt token's attention of the layer
before each stage (head mean, over the tokens still present) and the hidden states entering the stage layer:
1. before decoder layer 3:  keep k = N/2: the top int(0.1k x 1.2) by attention, the rest by DivPrune (max-min cosine
   diversity) seeded with them;
2. before layer 6:  keep k/2: the top int(0.4 x k/2) by attention, the rest by DivPrune seeded with them;
3. before layer 15: keep k/4 by attention, preferring tokens among the first k/4 survivors in raster order (the
   official `indices < k` rule);
4. before layer 22: drop every visual token.
The layers (official `start_layer, img_sense_layer, rel_start_layer, end_layer` = 4, 7, 16, 23, 1-based) were
calibrated by the authors for Qwen2.5-VL-7B, Lingshu-7B's architecture (28 layers).

Deviations:
- Budget: the first stage keeps a fraction f instead of 1/2 (later stages still halve), with f chosen so the mean
  number of visual tokens per decoder layer equals FastV's at the same budget (plan.md §C3). BTP's own schedule
  keeps 27% of the visual tokens per layer on average, FastV at 10% keeps 16%.
- Qwen3-VL-8B (36 layers): the authors' calibration code is not released, so the stage layers are scaled by depth
  (round(l x 36 / 28) = 4, 8, 19, 28).
- Robust to tiny budgets: where the official code would call `len()` on a 0-d tensor (one attention token), the
  index list is kept a list.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

QWEN25_STAGE_LAYERS = (3, 6, 15, 22)  # first decoder layer (0-based) without the pruned tokens, per stage
QWEN25_DEPTH = 28


def btp_layers(n_layers: int) -> tuple[int, int, int, int]:
    if n_layers == QWEN25_DEPTH:
        return QWEN25_STAGE_LAYERS
    return tuple(int(round(layer * n_layers / QWEN25_DEPTH)) for layer in QWEN25_STAGE_LAYERS)


def btp_first_fraction(budget: float, n_layers: int, first: int = 2) -> float:
    """Keep fraction of stage 1 that makes the mean visual tokens per layer equal FastV's at `budget`."""
    l1, l2, l3, l4 = btp_layers(n_layers)
    target = first + (n_layers - first) * budget
    f = (target - l1) / ((l2 - l1) + (l3 - l2) / 2 + (l4 - l3) / 4)
    if f <= 0:
        raise ValueError(f"budget {budget} is below BTP's {l1} full layers on a {n_layers}-layer decoder")
    return min(1.0, f)


def div_prune(hidden: torch.Tensor, k: int, seed: list[int] | None = None) -> list[int]:
    """Greedy max-min cosine-distance selection of k more tokens, after the `seed` tokens (returned first)."""
    x = F.normalize(hidden.float(), p=2, dim=1)
    dist = 1 - x @ x.T
    dist.fill_diagonal_(float("inf"))
    chosen = list(seed or [])
    if chosen:
        dist[:, chosen] = float("-inf")
    for _ in range(k):
        if not chosen:
            pick = int(dist.min(dim=1).values.argmax())
        else:
            pick = int(dist[chosen, :].min(dim=0).values.argmax())
        chosen.append(pick)
        dist[:, pick] = float("-inf")
    return chosen


def _top(attn: torch.Tensor, count: int) -> list[int]:
    return torch.topk(attn, count).indices.reshape(-1).tolist() if count > 0 else []


def stage_balanced(attn: np.ndarray, hidden: np.ndarray, k: int, attn_share: float, extend: float) -> np.ndarray:
    """Stages 1-2 on the surviving tokens (arrays already restricted to them, in raster order): indices kept."""
    a = torch.as_tensor(attn, dtype=torch.float32)
    attn_idx = _top(a, int(int(k * attn_share) * extend))
    rest = k - len(attn_idx)
    idx = div_prune(torch.as_tensor(hidden), rest, attn_idx) if rest > 0 else attn_idx
    return np.sort(np.asarray(idx, dtype=np.int64))


def stage_attention(attn: np.ndarray, k: int) -> np.ndarray:
    """Stage 3: top int(1.2k) by attention, those among the first k survivors first, then the others by attention."""
    top = torch.as_tensor(_top(torch.as_tensor(attn, dtype=torch.float32), int(k * 1.2)), dtype=torch.long)
    pre, post = top[top < k], top[top >= k]
    need = k - pre.shape[0]
    idx = torch.cat((pre, post[:need])) if need > 0 else pre
    return np.sort(idx.numpy().astype(np.int64))
