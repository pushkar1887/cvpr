"""The remedy's selectors and merge plans (medprune/remedy.py, plan.md E40-E41)."""

import json

import numpy as np
import pytest

from medprune.remedy import (background_slot, curve_order, fastv_matched_count, hilbert_index, merge_plan, pool_plan,
                             select_stratified)


def grid_rc(h: int, w: int) -> np.ndarray:
    return np.stack(np.divmod(np.arange(h * w), w), axis=1)


@pytest.mark.parametrize("side", [2, 4, 8, 16])
def test_hilbert_visits_every_cell_through_neighbours(side):
    rc = grid_rc(side, side)
    d = hilbert_index(rc, side)
    assert sorted(d) == list(range(side * side))
    path = rc[np.argsort(d)]
    assert np.all(np.abs(np.diff(path, axis=0)).sum(axis=1) == 1)  # each step moves to an adjacent cell


def test_curve_order_handles_non_square_grids():
    rc = grid_rc(5, 7)
    order = curve_order(rc)
    assert sorted(order) == list(range(35))


def test_stratified_puts_one_token_in_each_quadrant():
    rc = grid_rc(8, 8)
    for seed in range(20):
        keep = select_stratified(rc, 4, np.random.default_rng(seed))
        quadrants = {(r // 4, c // 4) for r, c in rc[keep]}
        assert len(keep) == 4 and len(quadrants) == 4  # Hilbert strata of 16 tokens are the four quadrants


def test_stratified_stays_inside_allowed_and_respects_exclude():
    rc = grid_rc(8, 8)
    allowed = rc[:, 1] >= 3
    keep = select_stratified(rc, 6, np.random.default_rng(0), allowed, exclude=int(np.flatnonzero(allowed)[0]))
    assert len(keep) == 6 and allowed[keep].all() and np.flatnonzero(allowed)[0] not in keep
    small = np.zeros(64, dtype=bool)
    small[[5, 9]] = True
    keep = select_stratified(rc, 6, np.random.default_rng(0), small)  # too few allowed: all of them, rest outside
    assert len(set(keep)) == 6 and {5, 9} <= set(keep)


def test_stratified_avoids_the_fixed_grid_aliasing():
    from medprune.selection import select_uniform

    rc = grid_rc(8, 8)
    fixed_cols = set(rc[select_uniform(64, 6), 1])
    cols = [set(rc[select_stratified(rc, 6, np.random.default_rng(s)), 1]) for s in range(50)]
    assert len(set().union(*cols)) == 8 and len(fixed_cols) < 8  # every column is reachable; the fixed picks are not


def test_fastv_matched_count():
    assert fastv_matched_count(64, 6, 28) == 10  # Lingshu RIN: (2 * 64 + 26 * 6) / 28 = 10.1
    assert fastv_matched_count(64, 6, 36) == 9  # Qwen3-VL RIN: (2 * 64 + 34 * 6) / 36 = 9.2
    assert fastv_matched_count(324, 32, 28) == 53  # Lingshu SLAKE 18 x 18; VScan keeps 53.5 per layer
    assert fastv_matched_count(1, 1, 28) == 1


def test_merge_plan_without_gate_equals_vscans_merge():
    import torch

    from medprune.baselines.vscan import merge_weights

    rng = np.random.default_rng(0)
    emb = rng.normal(size=(40, 16)).astype(np.float32)
    keep = np.sort(rng.choice(40, 7, replace=False))
    rows, w = merge_plan(emb, keep)
    assert np.array_equal(rows, keep)
    assert np.allclose(w, merge_weights(torch.as_tensor(emb), torch.as_tensor(keep)))


def test_foreground_merge_discards_background_and_bgsum_collects_it():
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(30, 8))
    allowed = np.arange(30) >= 10  # tokens 0-9 are background
    keep = np.array([12, 18, 25])
    rows, w = merge_plan(emb, keep, allowed)
    assert np.allclose(w.sum(axis=1), 1) and np.all(w[:, :10] == 0)
    assert np.all((w[:, 10:] > 0).sum(axis=0) == 1)  # each foreground token sits in exactly one kept row
    slot = background_slot(emb, allowed)
    assert 0 <= slot < 10
    keep = np.sort(np.append(keep, slot))
    rows, w = merge_plan(emb, keep, allowed, summary=slot)
    r = int(np.searchsorted(rows, slot))
    assert np.allclose(w[r, :10], 0.1) and np.all(w[r, 10:] == 0)
    others = np.delete(w, r, axis=0)
    assert np.all(others[:, :10] == 0) and np.all((others[:, 10:] > 0).sum(axis=0) == 1)


def test_background_slot_is_none_without_background_or_foreground():
    emb = np.ones((4, 3))
    assert background_slot(emb, np.ones(4, dtype=bool)) is None
    assert background_slot(emb, np.zeros(4, dtype=bool)) is None


@pytest.mark.parametrize("h,w,k", [(8, 8, 6), (8, 8, 1), (18, 18, 32), (18, 25, 45), (5, 3, 15), (36, 20, 72)])
def test_pool_plan_partitions_the_grid_into_k_blocks(h, w, k):
    rc = grid_rc(h, w)
    keep, (rows, weights) = pool_plan(rc, (h, w), k)
    assert len(keep) == k == len(set(keep)) and np.array_equal(rows, keep)
    assert np.allclose(weights.sum(axis=1), 1)
    assert np.all((weights > 0).sum(axis=0) == 1)  # every token in exactly one block
    sizes = (weights > 0).sum(axis=1)
    assert sizes.max() <= 2 * sizes.min() + 4  # integer grids: rows of 3 vs 4 and columns of 2 vs 3 are unavoidable
    for row, center in enumerate(keep):
        assert weights[row, center] > 0  # the pooled token sits inside its own block


def _bundle(tmp_path):
    from medprune.data import Sample, write_manifest
    from PIL import Image

    bundle = tmp_path / "bundle"
    (bundle / "images").mkdir(parents=True)
    samples = []
    for i in range(3):
        Image.fromarray(np.full((32, 32, 3), 60 * i, np.uint8)).save(bundle / "images" / f"{i}.png")
        samples.append(Sample(qid=f"toy/{i}", dataset="slake_en", image_id=f"toy/{i}", image_path=f"images/{i}.png",
                              question="Is it?", answer="yes", answer_type="yesno", prompt_style="close"))
    write_manifest(samples, bundle, "slake_en")
    return bundle


def test_remedy_methods_dry_run(tmp_path):
    from medprune.run_eval import RunSpec, run_many

    bundle, out = _bundle(tmp_path), tmp_path / "results"
    names = ("jitter_merge", "gate_jitter_merge", "gate_jitter_bgsum", "pool", "gate_jitter_merge_eq",
             "gate_jitter_bgsum_eq")
    specs = [RunSpec("lingshu7b", "slake_en", m, 0.25) for m in names]
    run_many(specs, bundle, out, {"lingshu7b": {"slake": 1.0, "text_only": 0.5}}, dry_run=True)
    rows = {s.method: [json.loads(l) for l in open(out / "runs" / f"{s.run_id('dryrun')}.jsonl")] for s in specs}
    for name in names:
        n = rows[name][0]["n_vis"]
        k = int(round(0.25 * n))
        expect = fastv_matched_count(n, k, 28) if name.endswith("_eq") else k
        assert all(r["k_keep"] == expect for r in rows[name]), name
    # without a gate file the gate keeps everything: the gated rows choose the same tokens as jitter_merge
    assert [r["keep_idx"] for r in rows["jitter_merge"]] == [r["keep_idx"] for r in rows["gate_jitter_merge"]]
    assert [r["keep_idx"] for r in rows["gate_jitter_merge"]] == [r["keep_idx"] for r in rows["gate_jitter_bgsum"]]
