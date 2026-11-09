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
- `slic_plan` (E46): SLIC clusters in token space (content + position), seeded by the strata.
- `kmeans_plan` (E43): k-means++ + spherical Lloyd clusters of the foreground, each kept at its medoid with the
  cluster mean (minimizes the merge's averaging error, the residual once the size term restores the attention mass).
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


def kmeans_plan(emb: np.ndarray, rc: np.ndarray, k: int, rng: np.random.Generator, allowed: np.ndarray | None = None,
                iters: int = 2, device: str | None = None) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """E43 (2A): k clusters of the `allowed` tokens chosen to minimize the averaging error of the merge.

    With the size term the merged cluster's attention mass is exact for duplicates, and the residual error of its value
    term is bounded (Cauchy-Schwarz, values linear in the input) by sqrt(sum_C ||x_j - mean_C||^2): the k-means
    objective. Seeding is k-means++ (D^2 sampling on unit vectors, ||x - c||^2 = 2 - 2 cos; E[cost] <= 8(ln k + 2) OPT,
    Arthur & Vassilvitskii, SODA 2007), followed by `iters` spherical Lloyd steps (each never increases the cost).
    Each cluster is kept at its medoid (the member most similar to the cluster mean) and carries the uniform mean of its
    members; tokens outside `allowed` are discarded. Exactly min(k, n) tokens are kept: empty clusters are refilled by
    splitting off the worst-fitting member of the largest cluster. When fewer than k tokens are allowed, all of them
    are kept and the rest is topped up from outside by Hilbert strata, unmerged. Computed on the Gram matrix (one
    n x n product) on `device` (CUDA when available). Returns (keep, (rows, weights [k, n])).
    """
    import torch

    n = len(emb)
    k = max(1, min(int(k), n))
    inside = np.ones(n, dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)
    cand = np.flatnonzero(inside)
    if cand.size <= k:  # nothing to cluster: every allowed token kept as itself, topped up by strata, unmerged
        keep = select_stratified(rc, k, rng, inside)
        weights = np.zeros((keep.size, n))
        weights[np.arange(keep.size), keep] = 1.0
        return keep, (keep, weights)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(_unit(emb[cand]), dtype=torch.float32, device=dev)
    gram = (x @ x.T).clamp_(-1.0, 1.0)  # cosine similarities, [m, m]
    m = cand.size
    # k-means++ seeding: first centre uniform, then each next centre with probability proportional to D^2
    first = int(rng.integers(m))
    centres = [first]
    d2 = (2 - 2 * gram[first]).clamp_(min=0)
    taken = torch.zeros(m, dtype=torch.bool, device=dev)
    taken[first] = True
    draws = rng.random(k)
    for t in range(1, k):
        w = torch.where(taken, torch.zeros_like(d2), d2)
        total = float(w.sum())
        if total <= 0:  # remaining points coincide with centres: uniform over the untaken ones
            w = (~taken).float()
            total = float(w.sum())
        c = int(torch.searchsorted(torch.cumsum(w, 0), torch.tensor(draws[t] * total, device=dev)).clamp(max=m - 1))
        while taken[c]:  # float edge case at the CDF boundary
            c = (c + 1) % m
        centres.append(c)
        taken[c] = True
        d2 = torch.minimum(d2, (2 - 2 * gram[c]).clamp_(min=0))
    # spherical Lloyd on the Gram matrix: a point's cosine to a cluster mean is its summed Gram column / ||sum||
    member = torch.zeros(m, k, device=dev)
    member[torch.arange(m, device=dev), gram[:, centres].argmax(dim=1)] = 1.0
    for _ in range(iters):
        sizes = member.sum(dim=0)
        sim = gram @ member  # [m, k]: sum of similarities to each cluster's members
        norm = torch.sqrt(torch.clamp((member * sim).sum(dim=0), min=1e-12))  # ||sum of members|| per cluster
        score = torch.where(sizes > 0, sim / norm, torch.full_like(sim, -2.0))  # cosine to the cluster mean
        new = torch.zeros_like(member)
        new[torch.arange(m, device=dev), score.argmax(dim=1)] = 1.0
        member = new
    sizes = member.sum(dim=0)
    sim = gram @ member
    norm = torch.sqrt(torch.clamp((member * sim).sum(dim=0), min=1e-12))
    fit = (sim / norm)  # cosine of each point to each cluster mean
    label = member.argmax(dim=1)
    while int((sizes == 0).sum()):  # refill an empty cluster with the worst-fitting member of the largest cluster
        empty, big = int((sizes == 0).nonzero()[0]), int(sizes.argmax())
        in_big = (label == big).nonzero().flatten()
        worst = int(in_big[fit[in_big, big].argmin()])
        label[worst] = empty
        sizes[big] -= 1
        sizes[empty] += 1
    label = label.cpu().numpy()
    fit = fit.cpu().numpy()
    keep, groups = [], []
    for j in range(k):
        members = np.flatnonzero(label == j)
        medoid = members[np.argmax(fit[members, j])] if members.size > 1 else members[0]
        keep.append(int(cand[medoid]))
        groups.append(cand[members])
    order = np.argsort(keep)
    keep = np.asarray(keep, dtype=np.int64)[order]
    weights = np.zeros((k, n))
    for row, j in enumerate(order):
        weights[row, groups[j]] = 1.0 / groups[j].size
    return keep, (keep, weights)


def slic_plan(emb: np.ndarray, rc: np.ndarray, k: int, rng: np.random.Generator, allowed: np.ndarray | None = None,
              iters: int = 3, device: str | None = None) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """E46: k compact, content-coherent merge clusters of the `allowed` tokens (SLIC, Achanta et al., TPAMI 2012, in
    token space) — between block pooling (spatial only; lost anatomy) and k-means (content only; lost layout).

    Seeds are the recipe's coverage (`select_stratified`: one random token per Hilbert stratum). Lloyd steps minimize
    d^2 = ||u_i - m_j||^2 / s_f^2 + ||p_i - c_j||^2 / S^2 (u: unit embeddings, p: grid positions, m_j / c_j: the cluster's
    mean embedding / position) with S = sqrt(n_allowed / k), the expected cluster spacing, and s_f^2 = the median squared
    feature distance of each token to the seed nearest in space: both terms on their natural scale, no tuned weight.
    Each cluster is kept at its medoid under d and carries the uniform mean of its members; tokens outside `allowed`
    are discarded; exactly min(k, n) tokens are kept (empty clusters refilled as in `kmeans_plan`). Returns
    (keep, (rows, weights [k, n])).
    """
    import torch

    n = len(emb)
    k = max(1, min(int(k), n))
    inside = np.ones(n, dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)
    cand = np.flatnonzero(inside)
    if cand.size <= k:
        keep = select_stratified(rc, k, rng, inside)
        weights = np.zeros((keep.size, n))
        weights[np.arange(keep.size), keep] = 1.0
        return keep, (keep, weights)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seeds = select_stratified(rc, k, rng, inside)
    local = np.searchsorted(cand, seeds)
    x = torch.as_tensor(_unit(emb[cand]), dtype=torch.float32, device=dev)
    p = torch.as_tensor(np.asarray(rc)[cand], dtype=torch.float32, device=dev)
    m, c = x[local].clone(), p[local].clone()
    spacing2 = float(cand.size) / k  # S^2
    pd = torch.cdist(p, c) ** 2  # [n_allowed, k]
    nearest = pd.argmin(dim=1)
    s_f2 = float(torch.median(((x - m[nearest]) ** 2).sum(dim=1)).clamp(min=1e-6))

    def distance(m, c):
        fd = (x * x).sum(1, keepdim=True) + (m * m).sum(1)[None, :] - 2 * x @ m.T
        return fd.clamp(min=0) / s_f2 + torch.cdist(p, c) ** 2 / spacing2

    label = distance(m, c).argmin(dim=1)
    for _ in range(iters):
        onehot = torch.zeros(cand.size, k, device=dev)
        onehot[torch.arange(cand.size, device=dev), label] = 1.0
        sizes = onehot.sum(dim=0)
        filled = sizes > 0
        m = torch.where(filled[:, None], (onehot.T @ x) / sizes.clamp(min=1)[:, None], m)
        c = torch.where(filled[:, None], (onehot.T @ p) / sizes.clamp(min=1)[:, None], c)
        label = distance(m, c).argmin(dim=1)
    d = distance(m, c)
    sizes = torch.bincount(label, minlength=k).float()
    while int((sizes == 0).sum()):  # refill an empty cluster with the worst-fitting member of the largest cluster
        empty, big = int((sizes == 0).nonzero()[0]), int(sizes.argmax())
        in_big = (label == big).nonzero().flatten()
        worst = int(in_big[d[in_big, big].argmax()])
        label[worst] = empty
        sizes[big] -= 1
        sizes[empty] += 1
    label, d = label.cpu().numpy(), d.cpu().numpy()
    keep, groups = [], []
    for j in range(k):
        members = np.flatnonzero(label == j)
        medoid = members[np.argmin(d[members, j])]
        keep.append(int(cand[medoid]))
        groups.append(cand[members])
    order = np.argsort(keep)
    keep = np.asarray(keep, dtype=np.int64)[order]
    weights = np.zeros((k, n))
    for row, j in enumerate(order):
        weights[row, groups[j]] = 1.0 / groups[j].size
    return keep, (keep, weights)


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
