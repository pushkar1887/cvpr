"""The anatomy gate in front of every published pruner (plan.md E49, gate_<method>).

One rule for all (medprune/baselines/__init__.py): only the gate's foreground tokens are candidates, and when the
foreground holds no more tokens than a selection keeps, all of it is kept and the method's own ranking fills the rest
from the background. Three properties per pruner:
1. With every token in the foreground the gated variant is the ungated method, token for token (and merge for merge).
2. With a partial gate and enough foreground, exactly the budget is kept and every kept token is foreground.
3. With a foreground smaller than the budget, all of it is kept, topped up to the budget.
"""

import json

import numpy as np
import pytest

from medprune.baselines import keep_pool
from medprune.baselines.hiprune import select_hiprune
from medprune.baselines.prunesid import select_prunesid
from medprune.baselines.vispruner import select_vispruner
from medprune.baselines.vscan import gated_window_selection, stage1, window_selection
from medprune.remedy import gated_pool_plan, pool_plan
from medprune.run_eval import _gate_stage

GATED = ("sparsevlm", "btp", "vispruner", "prunesid", "hiprune", "feather", "vscan", "transprune", "pool")


def _rc(h, w):
    return np.stack(np.divmod(np.arange(h * w), w), axis=1)


def _disc(h, w, radius):
    """A centred disc of foreground (anatomy) on background, as a boolean [h * w] gate."""
    rc = _rc(h, w)
    return ((rc[:, 0] - (h - 1) / 2) ** 2 + (rc[:, 1] - (w - 1) / 2) ** 2) <= radius ** 2


def _cues(n, d=16, seed=0):
    g = np.random.default_rng(seed)
    return g.random(n) + 1e-3, g.random(n) + 1e-3, g.standard_normal((n, d))


# --------------------------------------------------------------------------------------------- the shared helper

def test_keep_pool_keeps_the_pool_and_tops_up_by_the_order():
    assert keep_pool(np.array([4, 1]), 4, np.array([1, 7, 4, 0, 3])).tolist() == [0, 1, 4, 7]
    assert keep_pool(np.array([2, 3]), 2, np.array([0, 1])).tolist() == [2, 3]


# ------------------------------------------------------------------------------------------------- pre-LLM pruners

@pytest.mark.parametrize("h,w,k", [(8, 8, 7), (18, 18, 32), (16, 20, 20)])
def test_vispruner_gate_of_every_token_is_vispruner(h, w, k):
    imp, _, feats = _cues(h * w)
    everything = np.ones(h * w, dtype=bool)
    assert np.array_equal(select_vispruner(imp, feats, k, allowed=everything), select_vispruner(imp, feats, k))


@pytest.mark.parametrize("h,w,k", [(8, 8, 7), (18, 18, 32), (16, 20, 20)])
def test_prunesid_gate_of_every_token_is_prunesid(h, w, k):
    _, _, feats = _cues(h * w, d=32)
    everything = np.ones(h * w, dtype=bool)
    assert np.array_equal(select_prunesid(feats, k, 3, everything), select_prunesid(feats, k, 3))


@pytest.mark.parametrize("h,w,k", [(8, 8, 7), (18, 18, 32), (16, 20, 20)])
def test_hiprune_gate_of_every_token_is_hiprune(h, w, k):
    shallow, deep, _ = _cues(h * w)
    everything = np.ones(h * w, dtype=bool)
    assert np.array_equal(select_hiprune(shallow, deep, (h, w), k, allowed=everything),
                          select_hiprune(shallow, deep, (h, w), k))


@pytest.mark.parametrize("h,w,k", [(8, 8, 8), (18, 18, 33), (16, 20, 20), (3, 6, 6)])
def test_vscan_stage1_gate_of_every_token_is_vscan(h, w, k):
    local, glob, emb = _cues(h * w)
    everything = np.ones(h * w, dtype=bool)
    kept, (rows, weights) = stage1(local, glob, emb, (h, w), k, everything)
    kept0, (rows0, weights0) = stage1(local, glob, emb, (h, w), k)
    assert np.array_equal(kept, kept0) and np.array_equal(rows, rows0)
    np.testing.assert_allclose(weights, weights0, atol=1e-12)


@pytest.mark.parametrize("h,w,k", [(8, 8, 4), (18, 18, 16), (16, 20, 10)])
def test_gated_window_selection_of_every_token_is_the_official_scan(h, w, k):
    import torch

    attn, _, _ = _cues(h * w)
    official = window_selection(torch.as_tensor(attn, dtype=torch.float32), k, h, w).numpy()
    assert np.array_equal(np.sort(gated_window_selection(attn, k, h, w, np.ones(h * w, dtype=bool))),
                          np.sort(official.astype(np.int64)))


@pytest.mark.parametrize("h,w,k", [(8, 8, 7), (18, 18, 32), (16, 20, 20), (4, 4, 3)])
def test_pool_gate_of_every_token_is_block_pooling(h, w, k):
    rc = _rc(h, w)
    keep, (rows, weights) = gated_pool_plan(rc, k, np.ones(h * w, dtype=bool))
    keep0, (rows0, weights0) = pool_plan(rc, (h, w), k)
    assert np.array_equal(keep, keep0) and np.array_equal(rows, rows0)
    np.testing.assert_allclose(weights, weights0)


def _pre_llm(name, h, w, k, allowed):
    imp, deep, emb = _cues(h * w, d=32)
    if name == "vispruner":
        return select_vispruner(imp, emb, k, allowed=allowed), None
    if name == "prunesid":
        return select_prunesid(emb, k, 3, allowed), None
    if name == "hiprune":
        return select_hiprune(imp, deep, (h, w), k, allowed=allowed), None
    if name == "vscan":
        return stage1(imp, deep, emb, (h, w), k, allowed)
    if name == "pool":
        return gated_pool_plan(_rc(h, w), k, allowed)
    raise KeyError(name)


@pytest.mark.parametrize("name", ["vispruner", "prunesid", "hiprune", "vscan", "pool"])
@pytest.mark.parametrize("h,w,k,radius", [(18, 18, 32, 6.5), (8, 8, 7, 2.6), (16, 20, 20, 7.0)])
def test_pre_llm_gate_keeps_the_budget_inside_the_foreground(name, h, w, k, radius):
    allowed = _disc(h, w, radius)
    assert allowed.sum() > k and not allowed.all()
    kept, merge = _pre_llm(name, h, w, k, allowed)
    assert kept.size == k == np.unique(kept).size
    assert allowed[kept].all()
    if merge is not None:  # merged rows read foreground only: background leaves unmerged
        rows, weights = merge
        assert np.array_equal(rows, kept)
        assert np.all(weights[:, ~allowed] == 0)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)


@pytest.mark.parametrize("name", ["vispruner", "prunesid", "hiprune", "vscan", "pool"])
def test_pre_llm_gate_smaller_than_the_budget_keeps_all_of_it(name):
    h, w, k = 18, 18, 32
    allowed = _disc(h, w, 2.5)
    assert 0 < allowed.sum() < k
    kept, merge = _pre_llm(name, h, w, k, allowed)
    assert kept.size == k == np.unique(kept).size
    assert np.isin(np.flatnonzero(allowed), kept).all()
    assert merge is None


def test_gated_pool_carries_the_foreground_mean_of_each_block():
    h, w, k = 18, 18, 12
    allowed = _disc(h, w, 6.0)
    keep, (rows, weights) = gated_pool_plan(_rc(h, w), k, allowed)
    assert keep.size == k and allowed[keep].all()
    covered = weights.sum(axis=0) > 0
    assert np.array_equal(covered, allowed)  # every foreground token is in exactly one block, no background
    assert np.all((weights > 0).sum(axis=0)[allowed] == 1)


# -------------------------------------------------------------------------------------------- in-LLM stage wrapper

def _pick(scores, survivors, count):
    alive = np.flatnonzero(survivors)
    return np.sort(alive[np.argsort(-scores[alive], kind="stable")[:count]])


def test_gate_stage_without_a_gate_is_the_stage():
    scores = np.random.default_rng(0).random(30)
    survivors = np.ones(30, dtype=bool)
    survivors[::4] = False
    for allowed in (None, np.ones(30, dtype=bool)):
        assert np.array_equal(_gate_stage(_pick, allowed, 9)(scores, survivors), _pick(scores, survivors, 9))


def test_gate_stage_picks_among_surviving_foreground_then_fills_from_background():
    scores = np.arange(30, dtype=float)[::-1].copy()  # token 0 scores highest
    survivors = np.ones(30, dtype=bool)
    allowed = np.zeros(30, dtype=bool)
    allowed[10:20] = True
    assert _gate_stage(_pick, allowed, 4)(scores, survivors).tolist() == [10, 11, 12, 13]
    assert _gate_stage(_pick, allowed, 13)(scores, survivors).tolist() == [0, 1, 2, *range(10, 20)]


def test_gate_stage_passes_extra_arguments_through():
    hidden = np.random.default_rng(1).standard_normal((12, 4))
    seen = {}

    def pick(scores, survivors, count, h):
        seen["h"] = h
        return _pick(scores, survivors, count)

    allowed = np.arange(12) % 2 == 0
    out = _gate_stage(pick, allowed, 3)(np.arange(12.0), np.ones(12, dtype=bool), hidden)
    assert out.tolist() == [6, 8, 10] and seen["h"] is hidden


# ----------------------------------------------------------------------------------------------- end to end (fake)

def test_every_gated_pruner_with_an_all_foreground_gate_runs_as_the_pruner(tmp_path):
    """Dry run on the fake model: no gate masks, so the gate is all foreground; each gate_<m> must keep m's tokens."""
    from medprune.data import Sample, write_manifest
    from medprune.run_eval import METHODS, RunSpec, run_many
    from PIL import Image

    bundle, out = tmp_path / "bundle", tmp_path / "results"
    (bundle / "images").mkdir(parents=True)
    samples = []
    for i in range(3):
        Image.fromarray(np.full((32, 32, 3), 60 * i, np.uint8)).save(bundle / "images" / f"{i}.png")
        samples.append(Sample(qid=f"toy/{i}", dataset="slake_en", image_id=f"toy/{i}", image_path=f"images/{i}.png",
                              question="Is it?", answer="yes", answer_type="yesno", prompt_style="close"))
    write_manifest(samples, bundle, "slake_en")
    for m in GATED:
        assert METHODS[f"gate_{m}"].gate == "foreground" and METHODS[f"gate_{m}"].rule == METHODS[m].rule
        assert METHODS[f"gate_{m}"].mask_from == METHODS[m].mask_from
    specs = [RunSpec("lingshu7b", "slake_en", m, 0.25) for name in GATED for m in (name, f"gate_{name}")]
    run_many(specs, bundle, out, {"lingshu7b": {"slake": 1.0, "text_only": 0.5}}, dry_run=True)
    rows = {s.method: [json.loads(l) for l in open(out / "runs" / f"{s.run_id('dryrun')}.jsonl")] for s in specs}
    for m in GATED:
        a, b = rows[m], rows[f"gate_{m}"]
        assert len(a) == len(b) == 3, m
        assert [r["keep_idx"] for r in a] == [r["keep_idx"] for r in b], m
        assert [r.get("visual_tokens_per_layer") for r in a] == [r.get("visual_tokens_per_layer") for r in b], m
