"""FEATHER (Endo et al., ICCV 2025, "Feather the Throttle"): RoPE-free attention plus uniform sampling at an early
layer, then RoPE-free attention alone at a later layer.

No public code; implemented from the paper (§3.4 criteria, §4.2 final approach):
- Stage 1 after layer K = 8: criterion phi_-R + phi_uniform — tokens on a stride-3 grid (every third row and column
  of the token grid, "a small number of uniformly sampled tokens using a stride of three"), the rest of the stage
  budget by phi_-R = the last prompt token's attention computed without RoPE (head mean), keeping (1 - R) of N.
- Stage 2 after layer K = 16: phi_-R alone on the survivors, keeping (1 - R)^2 of them, i.e. (1 - R)^3 of N.

Budget: the stage-2 count is the run's budget, so (1 - R) = budget^(1/3) (46.4% then 10% at a 10% budget). FEATHER
cannot be token-matched to FastV at 10%: its first eight layers are unpruned, which alone exceed FastV's mean visual
tokens per layer; FLOPs are reported (medprune.latency). Layers follow the paper (absolute, 8 and 16), as for the
`fastv_k8` / `fastv_k16` controls; scores are read at layers 7 and 15 (K - 1, FastV's convention).
"""

from __future__ import annotations

import numpy as np

STAGE_LAYERS = (8, 16)  # first layers at which each stage's dropped tokens are masked
STRIDE = 3


def feather_fractions(budget: float) -> tuple[float, float]:
    """(stage-1, stage-2) keep fractions of N: (1 - R) and (1 - R)^3 with (1 - R)^3 = budget."""
    a = budget ** (1.0 / 3.0)
    return a, budget


def uniform_grid(rc: np.ndarray, stride: int = STRIDE) -> np.ndarray:
    """Indices of the tokens on every `stride`-th row and column of the token grid (rc [N, 2] = (row, col))."""
    return np.flatnonzero((rc[:, 0] % stride == 0) & (rc[:, 1] % stride == 0)).astype(np.int64)


def stage1_keep(scores: np.ndarray, survivors: np.ndarray, uniform: np.ndarray, count: int) -> np.ndarray:
    """phi_-R + phi_uniform: the uniform tokens (evenly thinned if they exceed the count), the rest by score."""
    alive = np.flatnonzero(survivors)
    count = max(1, min(count, alive.size))
    uni = uniform[np.isin(uniform, alive)]
    if uni.size > count:
        uni = uni[np.unique(np.floor(np.linspace(0, uni.size - 1, count) + 0.5).astype(int))]
    rest = alive[~np.isin(alive, uni)]
    order = rest[np.argsort(-np.asarray(scores, dtype=np.float64)[rest], kind="stable")]
    return np.sort(np.concatenate([uni, order[: count - uni.size]])).astype(np.int64)


def stage2_keep(scores: np.ndarray, survivors: np.ndarray, count: int) -> np.ndarray:
    alive = np.flatnonzero(survivors)
    count = max(1, min(count, alive.size))
    order = alive[np.argsort(-np.asarray(scores, dtype=np.float64)[alive], kind="stable")]
    return np.sort(order[:count]).astype(np.int64)
