"""The remedy's pieces (plan.md E40-E41): stratified coverage, foreground-only merging, one background summary token,
an average-pooling baseline, and FastV-matched token counts. Pure NumPy.

Selectors return sorted, unique indices into the visual tokens (raster order). Merge plans are (rows, weights [R, N]):
`attention_hook.merge_rows` replaces vision-output row rows[i] by weights[i] @ output before the LLM, so a kept token
carries the mean of itself and the tokens assigned to it (VScan's `token_merging`, scaling 1).

- `select_stratified`: candidates ordered along a Hilbert curve over the token grid and cut into k contiguous strata of
  near-equal size; one random token per stratum. Strata are spatially compact, so the keep-set covers the image like a
  grid without a fixed grid's aliasing (evenly spaced raster picks land on the same columns of every 8 x 8 image).
- `merge_plan`: every dropped foreground token joins its most cosine-similar kept foreground token; dropped background
  tokens are discarded (`allowed` given) or join the background summary token (`summary` given).
- `background_slot`: the background token most similar to the background's mean embedding; its row becomes the mean
  of every dropped background token.
- `pool_plan`: exactly k near-equal-area rectangular blocks; each block's mean sits at its central token.
- `fastv_matched_count`: tokens kept from layer 0 so the mean visual tokens per decoder layer equal FastV's at the same
  budget (all N tokens in the first `first` layers, k after) — VScan's budget (plan.md §C3).
"""

from __future__ import annotations

import numpy as np


def fastv_matched_count(n: int, k: int, n_layers: int, first: int = 2) -> int:
    """Round of FastV's mean visual tokens per layer, (first * n + (L - first) * k) / L, clipped to [1, n]."""
    mean = (first * n + (n_layers - first) * k) / n_layers
    return max(1, min(n, int(np.floor(mean + 0.5))))


def hilbert_index(rc: np.ndarray, side: int) -> np.ndarray:
    """Distance along the Hilbert curve of a side x side grid (side a power of two) for each (row, col) in rc."""
    y, x = rc[:, 0].astype(np.int64).copy(), rc[:, 1].astype(np.int64).copy()
    d = np.zeros(len(rc), dtype=np.int64)
    s = side // 2
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        turn = ry == 0
        flip = turn & (rx == 1)
        x[flip], y[flip] = side - 1 - x[flip], side - 1 - y[flip]
        x[turn], y[turn] = y[turn].copy(), x[turn].copy()
        s //= 2
    return d


def curve_order(rc: np.ndarray) -> np.ndarray:
    """Token indices sorted along the Hilbert curve that covers the token grid."""
    side = 1
    while side < int(rc.max()) + 1:
        side *= 2
    return np.argsort(hilbert_index(rc, side), kind="stable")


def _one_per_stratum(order: np.ndarray, candidates: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    if m <= 0 or candidates.size == 0:
        return candidates[:0]
    if m >= candidates.size:
        return candidates.copy()
    along = order[np.isin(order, candidates)]
    return np.array([rng.choice(stratum) for stratum in np.array_split(along, m)], dtype=np.int64)


def select_stratified(rc: np.ndarray, k: int, rng: np.random.Generator, allowed: np.ndarray | None = None,
                      exclude: int | None = None) -> np.ndarray:
    """k tokens, one random token per Hilbert stratum of the `allowed` tokens (all when None); topped up the same way
    from outside `allowed` when it holds fewer than k. `exclude` (a token index) is never chosen."""
    n = len(rc)
    inside = np.ones(n, dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool).copy()
    free = np.ones(n, dtype=bool)
    if exclude is not None:
        free[exclude] = False
    order = curve_order(rc)
    chosen = _one_per_stratum(order, np.flatnonzero(inside & free), k, rng)
    if chosen.size < k:
        rest = _one_per_stratum(order, np.flatnonzero(~inside & free), k - chosen.size, rng)
        chosen = np.concatenate([chosen, rest])
    return np.sort(chosen).astype(np.int64)


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def background_slot(emb: np.ndarray, allowed: np.ndarray) -> int | None:
    """Token that will hold the background summary: the background token most cosine-similar to the background mean.
    None when the gate keeps everything (no background) or nothing (no foreground)."""
    allowed = np.asarray(allowed, dtype=bool)
    background = np.flatnonzero(~allowed)
    if background.size == 0 or not allowed.any():
        return None
    x = _unit(emb[background])
    return int(background[np.argmax(x @ x.mean(axis=0))])


def merge_plan(emb: np.ndarray, keep: np.ndarray, allowed: np.ndarray | None = None,
               summary: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(rows, weights [len(keep), N]) for `merge_rows`.

    allowed None: every dropped token joins its most cosine-similar kept token (= `baselines.vscan.merge_weights`).
    allowed given: only dropped foreground tokens move, to their most similar kept foreground token (any kept token if
    none is foreground); dropped background tokens are discarded, or averaged into `summary` (a kept background token).
    """
    n = len(emb)
    keep = np.sort(np.asarray(keep, dtype=np.int64))
    kept = np.zeros(n, dtype=bool)
    kept[keep] = True
    fg = np.ones(n, dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)
    weights = np.zeros((keep.size, n), dtype=np.float64)
    weights[np.arange(keep.size), keep] = 1.0
    targets = keep[fg[keep] & (keep != summary)]
    if targets.size == 0:
        targets = keep[keep != summary] if keep.size > 1 else keep
    movers = np.flatnonzero(~kept & fg)
    if movers.size:
        x = _unit(emb)
        nearest = targets[np.argmax(x[movers] @ x[targets].T, axis=1)]
        np.add.at(weights, (np.searchsorted(keep, nearest), movers), 1.0)
    if summary is not None:
        if not kept[summary]:
            raise ValueError("the summary token must be kept")
        weights[np.searchsorted(keep, summary), np.flatnonzero(~fg & ~kept)] = 1.0
    return keep, weights / weights.sum(axis=1, keepdims=True)


def pool_plan(rc: np.ndarray, grid_hw: tuple[int, int], k: int) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Average pooling to exactly k tokens: ceil-balanced horizontal bands, each split into equal-width blocks, so the
    k blocks have near-equal area. Returns (kept tokens, merge plan); a kept token is its block's central token and
    carries the block mean."""
    h, w = grid_hw
    n = len(rc)
    k = max(1, min(int(k), n))
    bands = int(np.clip(np.floor(np.sqrt(k * h / w) + 0.5), 1, min(h, k)))
    while -(-k // bands) > w:
        bands += 1
    base, extra = divmod(k, bands)
    counts = [base + (1 if b < extra else 0) for b in range(bands)]
    row_edges = np.floor(h * np.cumsum([0] + counts) / k + 0.5).astype(int)  # band height ∝ its block count
    if np.any(np.diff(row_edges) < 1):
        row_edges = np.floor(np.linspace(0, h, bands + 1) + 0.5).astype(int)
    centers, members = [], []
    for b in range(bands):
        r0, r1 = row_edges[b], row_edges[b + 1]
        m = counts[b]
        col_edges = np.floor(np.linspace(0, w, m + 1) + 0.5).astype(int)
        for j in range(m):
            c0, c1 = col_edges[j], col_edges[j + 1]
            block = np.flatnonzero((rc[:, 0] >= r0) & (rc[:, 0] < r1) & (rc[:, 1] >= c0) & (rc[:, 1] < c1))
            mid = ((r0 + r1 - 1) / 2, (c0 + c1 - 1) / 2)
            dist = (rc[block, 0] - mid[0]) ** 2 + (rc[block, 1] - mid[1]) ** 2
            centers.append(int(block[np.argmin(dist)]))
            members.append(block)
    order = np.argsort(centers)
    keep = np.asarray(centers, dtype=np.int64)[order]
    weights = np.zeros((k, n), dtype=np.float64)
    for row, i in enumerate(order):
        weights[row, members[i]] = 1.0 / members[i].size
    return keep, (keep, weights)
