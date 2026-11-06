"""Which visual tokens to keep — the method registry and pure NumPy selectors (plan.md §C3, §D3).

Every selector returns sorted, unique indices into the visual-token list (raster order) of length exactly k.
When a restriction (`allowed`) leaves fewer than k candidates, the selector keeps all of them and fills the rest
from outside the restriction (best score, or evenly spaced for score-free rules), so budgets are always equal.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MethodSpec:
    """How a method chooses tokens and where it acts inside the LLM.

    score_layer: decoder layer whose attention is read (K-1 for FastV); None if no scores are needed.
    mask_from:   first decoder layer at which dropped visual keys are masked; 0 = before the LLM.
    """

    name: str
    rule: str  # "none" | "random" | "uniform" | "topk" | "static" | "inverse" | "divprune" | "pdrop" |
    #            "sparsevlm" | "vispruner" | "prunesid" | "btp" | "visionzip" | "hiprune" | "cdpruner" | "feather" |
    #            "vscan" | "transprune" (medprune/baselines/)
    score: str | None = None  # None | "attn" (last prompt token) | "ropefree" | "text" (every token after the
    #                           image) | "raters" (SparseVLM text raters)
    score_layer: int | None = None
    mask_from: int | None = None
    gate: str | None = None  # None | "foreground" | "oracle"
    tau: float = 0.5
    bg_quota: float = 0.0
    two_pass: bool = False
    image: bool = True
    exclude_outliers: bool = False
    physical_gate: bool = False
    invert: bool = False  # keep the lowest scores instead (attention-informativeness control)
    profile: bool = False  # no pruning; log both readouts and sink scores at every layer (plan.md §A10 D1)


def _spec(name: str, rule: str, **kw) -> tuple[str, MethodSpec]:
    return name, MethodSpec(name=name, rule=rule, **kw)


METHODS: dict[str, MethodSpec] = dict(
    [
        _spec("full", "none"),
        _spec("text_only", "none", image=False),
        _spec("random_l2", "random", mask_from=2),
        _spec("random_l0", "random", mask_from=0),
        _spec("uniform_l2", "uniform", mask_from=2),
        _spec("fastv", "topk", score="attn", score_layer=1, mask_from=2),
        _spec("fastv_prellm", "topk", score="attn", score_layer=1, mask_from=0, two_pass=True),
        _spec("ropefree", "topk", score="ropefree", score_layer=1, mask_from=2),
        _spec("gate_fastv", "topk", score="attn", score_layer=1, mask_from=2, gate="foreground"),
        _spec("gate_ropefree", "topk", score="ropefree", score_layer=1, mask_from=2, gate="foreground"),
        _spec("gate_uniform", "uniform", mask_from=2, gate="foreground"),
        _spec("gate_random", "random", mask_from=2, gate="foreground"),
        _spec("fastv_k8", "topk", score="attn", score_layer=7, mask_from=8),
        _spec("fastv_k16", "topk", score="attn", score_layer=15, mask_from=16),
        _spec("static_map", "static", mask_from=2),
        _spec("inverse_gate", "inverse", mask_from=2, gate="foreground"),
        _spec("oracle_gate", "topk", score="attn", score_layer=1, mask_from=2, gate="oracle"),
        _spec("artifact_excl", "topk", score="attn", score_layer=1, mask_from=2, exclude_outliers=True),
        _spec("gate_quota3", "topk", score="attn", score_layer=1, mask_from=2, gate="foreground", bg_quota=0.03),
        _spec("gate_tau25", "topk", score="attn", score_layer=1, mask_from=2, gate="foreground", tau=0.25),
        _spec("gate_tau75", "topk", score="attn", score_layer=1, mask_from=2, gate="foreground", tau=0.75),
        _spec("divprune", "divprune", mask_from=0),
        _spec("gate_divprune", "divprune", mask_from=0, gate="foreground"),
        _spec("pdrop", "pdrop", score="attn", score_layer=1, mask_from=2),
        _spec("gate_pdrop", "pdrop", score="attn", score_layer=1, mask_from=2, gate="foreground"),
        _spec("gate_input_fastv", "topk", score="attn", score_layer=1, mask_from=2, gate="foreground",
              physical_gate=True),
        _spec("fastv_bottom", "topk", score="attn", score_layer=1, mask_from=2, invert=True),
        _spec("fastv_text", "topk", score="text", score_layer=1, mask_from=2),
        _spec("full_profile", "none", profile=True),
        # Published pruners, official selection code (medprune/baselines/, plan.md E13). Stage layers and budgets
        # come from the method's schedule at run time; mask_from here is the first layer any token is masked.
        _spec("sparsevlm", "sparsevlm", score="raters", score_layer=2, mask_from=3),
        _spec("vispruner", "vispruner", mask_from=0),
        _spec("prunesid", "prunesid", mask_from=0),
        _spec("btp", "btp", score="attn", mask_from=3),
        # SparseVLM ablation (plan.md E16, D-15): its text raters with one FastV-style stage at 10%, at FastV's layer
        # (raters vs last token at the same schedule) and at SparseVLM's first-stage layer (vs the 3-stage schedule).
        _spec("fastv_raters", "topk", score="raters", score_layer=1, mask_from=2),
        _spec("sparsevlm_single", "topk", score="raters", score_layer=2, mask_from=3),
        # Published pruners added Nov 7 (plan.md E14, G19): vision-encoder attention (VisionZip, HiPrune),
        # instruction-conditioned diversity without LLM attention (CDPruner), RoPE-free + uniform early stage (FEATHER),
        # encoder scans + mid-layer pruning (VScan), token transition + instruction attention (TransPrune).
        _spec("visionzip", "visionzip", mask_from=0),
        _spec("gate_visionzip", "visionzip", mask_from=0, gate="foreground"),
        _spec("hiprune", "hiprune", mask_from=0),
        _spec("cdpruner", "cdpruner", mask_from=0),
        _spec("gate_cdpruner", "cdpruner", mask_from=0, gate="foreground"),
        _spec("feather", "feather", score="ropefree", mask_from=8),
        _spec("vscan", "vscan", mask_from=0),
        _spec("transprune", "transprune", score="text", mask_from=7),
    ]
)


def keep_count(n: int, ratio: float) -> int:
    """Tokens kept at budget `ratio`: Python round(), as in FastV's code, clipped to [1, n]."""
    return max(1, min(n, int(round(ratio * n))))


def rng_for(seed: int, qid: str) -> np.random.Generator:
    """Per-question generator that is identical across processes (Python's str hash() is salted)."""
    digest = hashlib.sha256(f"{seed}:{qid}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _even_pick(candidates: np.ndarray, k: int) -> np.ndarray:
    """k evenly spaced members of a sorted candidate array (first one always included)."""
    if k <= 0 or candidates.size == 0:
        return candidates[:0]
    if k >= candidates.size:
        return candidates.copy()
    pos = np.floor(np.linspace(0, candidates.size - 1, k) + 0.5).astype(int)
    return candidates[np.unique(pos)]


def _finish(chosen: np.ndarray, n: int, k: int, fallback_order: np.ndarray) -> np.ndarray:
    """Top up `chosen` to k entries following `fallback_order`, then sort."""
    chosen = np.unique(chosen)
    if chosen.size < k:
        rest = fallback_order[~np.isin(fallback_order, chosen)]
        chosen = np.concatenate([chosen, rest[: k - chosen.size]])
    return np.sort(chosen[:k]).astype(np.int64)


def select_random(n: int, k: int, rng: np.random.Generator, allowed: np.ndarray | None = None) -> np.ndarray:
    """k tokens uniformly at random, among `allowed` tokens when given (topped up at random from the rest)."""
    if allowed is None:
        return np.sort(rng.choice(n, size=k, replace=False)).astype(np.int64)
    inside = np.flatnonzero(allowed)
    chosen = rng.choice(inside, size=min(k, inside.size), replace=False)
    outside = np.setdiff1d(np.arange(n), chosen)
    return _finish(chosen, n, k, rng.permutation(outside))


def select_uniform(n: int, k: int, allowed: np.ndarray | None = None) -> np.ndarray:
    """Evenly spaced in raster order, among `allowed` tokens when given."""
    everything = np.arange(n)
    candidates = everything if allowed is None else np.flatnonzero(allowed)
    chosen = _even_pick(candidates, k)
    outside = np.setdiff1d(everything, chosen)
    return _finish(chosen, n, k, _even_pick(outside, k - chosen.size) if chosen.size < k else outside)


def select_topk(scores: np.ndarray, k: int, allowed: np.ndarray | None = None) -> np.ndarray:
    """Highest scores, among `allowed` tokens when given (ties broken by position)."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    if allowed is None:
        return np.sort(order[:k]).astype(np.int64)
    inside = order[np.asarray(allowed, dtype=bool)[order]]
    return _finish(inside[:k], len(order), k, order)


def select_inverse(allowed: np.ndarray, k: int) -> np.ndarray:
    """Background-only control: k evenly spaced tokens among the tokens the gate rejects."""
    background = np.flatnonzero(~np.asarray(allowed, dtype=bool))
    chosen = _even_pick(background, k)
    foreground = np.flatnonzero(np.asarray(allowed, dtype=bool))
    return _finish(chosen, len(allowed), k, _even_pick(foreground, max(k - chosen.size, 0)))


def select_divprune(feats: np.ndarray, k: int, allowed: np.ndarray | None = None) -> np.ndarray:
    """DivPrune (Alvar et al., CVPR 2025, Alg. 1): greedy max-min cosine diversity.

    First pick: the token whose nearest neighbour is farthest. Then repeatedly add the candidate whose minimum
    distance to the selected set is largest.
    """
    x = np.asarray(feats, dtype=np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    dist = 1.0 - x @ x.T
    n = len(x)
    candidates = np.ones(n, dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool).copy()
    if candidates.sum() < k:  # restriction too tight: allow everything, keeping restricted tokens first
        first = np.flatnonzero(candidates)
        rest = select_divprune(feats, k, None)
        return _finish(first, n, k, rest)
    chosen: list[int] = []
    for i in range(k):
        if i == 0:
            nn = np.sort(dist, axis=0)[1] if n > 1 else np.zeros(1)  # distance to nearest other token
            score = nn
        else:
            score = dist[chosen].min(axis=0)
        score = np.where(candidates, score, -np.inf)
        score[chosen] = -np.inf
        chosen.append(int(np.argmax(score)))
    return np.sort(np.array(chosen, dtype=np.int64))


def sample_map(static_map: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear sample of a normalized map (R x R over [0,1]^2) at the centres of an H x W token grid."""
    r_res, c_res = static_map.shape
    h, w = grid_hw
    rows = ((np.arange(h) + 0.5) / h) * r_res - 0.5
    cols = ((np.arange(w) + 0.5) / w) * c_res - 0.5
    rr, cc = np.meshgrid(np.clip(rows, 0, r_res - 1), np.clip(cols, 0, c_res - 1), indexing="ij")
    r0, c0 = np.floor(rr).astype(int), np.floor(cc).astype(int)
    r1, c1 = np.minimum(r0 + 1, r_res - 1), np.minimum(c0 + 1, c_res - 1)
    fr, fc = rr - r0, cc - c0
    top = static_map[r0, c0] * (1 - fc) + static_map[r0, c1] * fc
    bottom = static_map[r1, c0] * (1 - fc) + static_map[r1, c1] * fc
    return (top * (1 - fr) + bottom * fr).reshape(-1)


def select_static(static_map: np.ndarray, grid_hw: tuple[int, int], k: int,
                  allowed: np.ndarray | None = None) -> np.ndarray:
    """Position-only control: rank tokens by a fixed map learned on another dataset."""
    return select_topk(sample_map(static_map, grid_hw), k, allowed)


def pdrop_schedule(rho: float, n_layers: int, first: int = 2) -> list[tuple[int, float]]:
    """PDrop stages token-matched to FastV at budget rho (plan.md §C3).

    Four stages start at `first` and every ceil((L - first) / 4) layers; keep fractions a, a/2, a/4, a/8 of N with a
    chosen so the mean number of visual tokens per layer equals FastV's (first + (L - first) * rho) / L.
    Returns [(boundary_layer, keep_fraction), ...].
    """
    step = math.ceil((n_layers - first) / 4)
    bounds = [first + i * step for i in range(4)]
    spans = [bounds[1] - bounds[0], bounds[2] - bounds[1], bounds[3] - bounds[2], n_layers - bounds[3]]
    weight = spans[0] + spans[1] / 2 + spans[2] / 4 + spans[3] / 8
    a = (n_layers - first) * rho / weight
    return [(b, min(1.0, a / 2**i)) for i, b in enumerate(bounds)]
