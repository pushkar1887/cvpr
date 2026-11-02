"""Uncertainty and mechanism diagnostics (plan.md §C7).

All intervals are cluster bootstraps over images: questions about the same image are not independent, and SLAKE's
test split has only 96 images.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def cluster_bootstrap_ci(values: np.ndarray, clusters: np.ndarray, B: int = 2000, seed: int = 0,
                         alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, low, high) of per-question values, resampling whole clusters (images) with replacement."""
    values = np.asarray(values, dtype=np.float64)
    _, cluster_index = np.unique(np.asarray(clusters), return_inverse=True)
    n_clusters = cluster_index.max() + 1
    sums = np.bincount(cluster_index, weights=values, minlength=n_clusters)
    counts = np.bincount(cluster_index, minlength=n_clusters).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clusters, size=(B, n_clusters))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(low), float(high)


def paired_delta_ci(correct_a: np.ndarray, correct_b: np.ndarray, clusters: np.ndarray, B: int = 2000,
                    seed: int = 0) -> tuple[float, float, float]:
    """Mean of A - B on identical questions with a cluster-bootstrap interval."""
    diff = np.asarray(correct_a, dtype=np.float64) - np.asarray(correct_b, dtype=np.float64)
    return cluster_bootstrap_ci(diff, clusters, B=B, seed=seed)


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """Two-sided exact McNemar p-value on discordant pairs."""
    a, b = np.asarray(correct_a, dtype=bool), np.asarray(correct_b, dtype=bool)
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    if only_a + only_b == 0:
        return 1.0
    return float(sps.binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue)


def background_retention_ratio(keep_idx: np.ndarray, background: np.ndarray) -> float:
    """P(background | kept) / P(background). Random selection gives 1 in expectation; NaN without contrast."""
    background = np.asarray(background, dtype=bool)
    share = background.mean()
    if share <= 0.0 or share >= 1.0:
        return float("nan")
    return float(background[np.asarray(keep_idx)].mean() / share)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a, b = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    return len(a & b) / len(a | b) if a | b else 1.0


def keep_map(keep_sets: list[np.ndarray], grids: list[tuple[int, int]], res: int = 16) -> np.ndarray:
    """Probability that the token at each normalized position is kept, res x res (nearest-cell resampling)."""
    total = np.zeros((res, res))
    for keep, (h, w) in zip(keep_sets, grids):
        kept = np.zeros(h * w, dtype=np.float64)
        kept[np.asarray(keep)] = 1.0
        rows = np.minimum(((np.arange(res) + 0.5) / res * h).astype(int), h - 1)
        cols = np.minimum(((np.arange(res) + 0.5) / res * w).astype(int), w - 1)
        total += kept.reshape(h, w)[np.ix_(rows, cols)]
    return total / max(len(keep_sets), 1)


def quadrant_concentration(kmap: np.ndarray, quadrant: str = "bottom_right") -> float:
    """Keep probability mass in a quadrant divided by the quadrant's area share (1 = uniform)."""
    h, w = kmap.shape
    rows = slice(h // 2, h) if quadrant.startswith("bottom") else slice(0, h // 2)
    cols = slice(w // 2, w) if quadrant.endswith("right") else slice(0, w // 2)
    total = kmap.sum()
    return float(kmap[rows, cols].sum() / total / 0.25) if total > 0 else float("nan")


def static_map_from_scores(scores: list[np.ndarray], grids: list[tuple[int, int]], res: int = 32) -> np.ndarray:
    """Mean rank-normalized score map on a res x res normalized grid (position-only control, plan.md §C7)."""
    total = np.zeros((res, res))
    for s, (h, w) in zip(scores, grids):
        ranks = sps.rankdata(np.asarray(s, dtype=np.float64))
        ranks = (ranks - 1) / max(len(ranks) - 1, 1)
        grid = ranks.reshape(h, w)
        rows = np.clip((np.arange(res) + 0.5) / res * h - 0.5, 0, h - 1)
        cols = np.clip((np.arange(res) + 0.5) / res * w - 0.5, 0, w - 1)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        r0, c0 = np.floor(rr).astype(int), np.floor(cc).astype(int)
        r1, c1 = np.minimum(r0 + 1, h - 1), np.minimum(c0 + 1, w - 1)
        fr, fc = rr - r0, cc - c0
        total += (grid[r0, c0] * (1 - fc) + grid[r0, c1] * fc) * (1 - fr) + \
                 (grid[r1, c0] * (1 - fc) + grid[r1, c1] * fc) * fr
    return total / max(len(scores), 1)


def dose_response(per_image_delta: np.ndarray, background_fraction: np.ndarray) -> dict[str, float]:
    """Spearman correlation between an image's background fraction and its Random - FastV accuracy gap."""
    rho, p = sps.spearmanr(background_fraction, per_image_delta)
    return {"spearman_rho": float(rho), "p_value": float(p), "n_images": int(len(per_image_delta))}


def keep_regression(df):
    """Logistic regression of 'kept' on position, background and token norms with image-clustered SEs.

    `df` columns: kept, row_norm, col_norm, background, vit_norm_z, llm_norm_z, dataset, image_id.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    model = smf.glm(
        "kept ~ bs(row_norm, df=4) + bs(col_norm, df=4) + background + vit_norm_z + llm_norm_z + C(dataset)",
        data=df,
        family=sm.families.Binomial(),
    )
    return model.fit(cov_type="cluster", cov_kwds={"groups": df["image_id"].astype("category").cat.codes})
