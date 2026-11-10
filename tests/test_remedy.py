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


def test_build_state_with_background_reserves_the_summary_slot():
    from medprune.data import Sample
    from medprune.run_eval import FakeAdapter, RunSpec, build_state
    from medprune.selection import METHODS

    adapter = FakeAdapter("lingshu7b")
    sample = Sample(qid="toy/0", dataset="slake_en", image_id="toy/0", image_path="x.png", question="Is it?",
                    answer="yes", answer_type="yesno", prompt_style="close")
    inputs = adapter.prepare(sample, image=object())
    gate = np.ones((4, 4), dtype=np.float32)
    gate[0, :] = 0.0  # the top row is background
    background = set(range(4))
    for name, k in (("gate_jitter_merge", 4), ("gate_jitter_bgsum", 4), ("gate_jitter_bgsum_eq", 5)):
        spec = RunSpec("lingshu7b", "slake_en", name, 0.25)
        state = build_state(spec, METHODS[name], sample, inputs, gate, None, adapter, None)
        keep = state.keep
        rows, w = state.merge
        assert len(keep) == k and np.array_equal(rows, keep), name
        bg_kept = [i for i in keep if i in background]
        if "bgsum" in name:
            assert len(bg_kept) == 1, name  # exactly one summary token, the rest foreground
            r = int(np.searchsorted(rows, bg_kept[0]))
            assert np.allclose(w[r, :4], 0.25) and np.all(w[r, 4:] == 0)
        else:
            assert not bg_kept and np.all(w[:, :4] == 0)  # background discarded
        assert np.allclose(w.sum(axis=1), 1)


def test_schedule_row_is_token_matched_and_nested(tmp_path):
    """E42b: k1 = keep_count(N, f1) from layer 0, k2 = keep_count(N, f2) of those from round(5L/7); the mean visual
    tokens per layer equals FastV's within one token (plan.md: f1 = 3(2 + 0.1(L - 2)) / (2c + L))."""
    from medprune.baselines.vscan import vscan_schedule
    from medprune.run_eval import RunSpec, run_many
    from medprune.selection import keep_count

    for n, layers in ((324, 28), (256, 36), (64, 28), (64, 36)):
        f1, cut, f2 = vscan_schedule(0.10, layers, layer=int(round(5 / 7 * layers)))
        assert cut == {28: 20, 36: 26}[layers]
        assert abs(f1 - 3 * (2 + 0.1 * (layers - 2)) / (2 * cut + layers)) < 2e-3  # stage-2 ratio 0.333 vs 1/3
        k1, k2 = keep_count(n, f1), keep_count(n, f2)
        fastv = (2 * n + (layers - 2) * keep_count(n, 0.10)) / layers
        assert abs((cut * k1 + (layers - cut) * k2) / layers - fastv) <= 1.0
    bundle, out = _bundle(tmp_path), tmp_path / "results"
    spec = RunSpec("lingshu7b", "slake_en", "gate_jitter_merge_sched", 0.25)
    run_many([spec], bundle, out, {"lingshu7b": {"slake": 1.0, "text_only": 0.5}}, dry_run=True)
    f1, _, f2 = vscan_schedule(0.25, 28, layer=20)
    for r in (json.loads(l) for l in open(out / "runs" / f"{spec.run_id('dryrun')}.jsonl")):
        (l1, s1), (l2, s2) = r["keep_stages"]
        assert (l1, l2) == (0, 20) and len(s1) == keep_count(16, f1) and len(s2) == keep_count(16, f2)
        assert set(s2) < set(s1) and r["keep_idx"] == s2  # stage 2 is a subset of stage 1


def test_size_term_counts_each_merged_cluster():
    """E42a keeps the reference's keep-set and merge exactly (same rng stream) and adds log of each cluster's size;
    with the gate, the sizes sum to the foreground (each foreground token in one cluster, background discarded)."""
    from medprune.data import Sample
    from medprune.run_eval import FakeAdapter, RunSpec, build_state
    from medprune.selection import METHODS

    adapter = FakeAdapter("lingshu7b")
    sample = Sample(qid="toy/1", dataset="slake_en", image_id="toy/1", image_path="x.png", question="Is it?",
                    answer="yes", answer_type="yesno", prompt_style="close")
    inputs = adapter.prepare(sample, image=object())
    gate = np.ones((4, 4), dtype=np.float32)
    gate[0, :] = 0.0
    states = {name: build_state(RunSpec("lingshu7b", "slake_en", name, 0.25), METHODS[name], sample, inputs, gate,
                                None, adapter, None) for name in ("gate_jitter_merge_eq", "gate_jitter_merge_pa_eq")}
    ref, pa = states["gate_jitter_merge_eq"], states["gate_jitter_merge_pa_eq"]
    assert np.array_equal(ref.keep, pa.keep) and np.array_equal(ref.merge[1], pa.merge[1])
    assert ref.key_bias is None and pa.key_bias is None  # the term is attached in run_one, after any merge
    pa.size_term_from_merge()
    rows, log_sizes = pa.key_bias
    sizes = np.exp(log_sizes)
    assert np.array_equal(rows, pa.keep) and pa.log_share
    np.testing.assert_allclose(sizes, (pa.merge[1] > 0).sum(axis=1), rtol=1e-6)
    assert round(float(sizes.sum())) == 12 and np.all(sizes >= 1)


def test_e42_rows_dry_run(tmp_path):
    from medprune.run_eval import RunSpec, run_many

    bundle, out = _bundle(tmp_path), tmp_path / "results"
    names = ("gate_jitter_merge_pa_eq", "gate_jitter_merge_sched", "gate_jitter_merge_sched_pa")
    specs = [RunSpec("lingshu7b", "slake_en", m, 0.10) for m in names]
    run_many(specs, bundle, out, {"lingshu7b": {"slake": 1.0, "text_only": 0.5}}, dry_run=True)
    rows = {s.method: [json.loads(l) for l in open(out / "runs" / f"{s.run_id('dryrun')}.jsonl")] for s in specs}
    assert all(r["k_keep"] == fastv_matched_count(16, 2, 28) and r["keep_stages"] is None
               for r in rows["gate_jitter_merge_pa_eq"])
    for name in names[1:]:
        assert all(len(r["keep_stages"]) == 2 for r in rows[name]), name
    # the size term does not change which tokens are kept
    assert [r["keep_stages"] for r in rows["gate_jitter_merge_sched"]] == \
        [r["keep_stages"] for r in rows["gate_jitter_merge_sched_pa"]]


# ------------------------------------------------------------------------------- E43 (2A): k-means merge clusters

def _distortion(emb: np.ndarray, weights: np.ndarray) -> float:
    """Sum over merged rows of ||x_j - row mean||^2 on unit vectors (the k-means objective of the merge)."""
    from medprune.remedy import _unit

    x = _unit(emb)
    total = 0.0
    for row in weights:
        members = np.flatnonzero(row > 0)
        total += float(((x[members] - x[members].mean(axis=0)) ** 2).sum())
    return total


def test_kmeans_plan_partitions_the_gate_into_exactly_k_clusters():
    from medprune.remedy import kmeans_plan

    rng = np.random.default_rng(3)
    rc = grid_rc(8, 8)
    emb = rng.normal(size=(64, 12))
    allowed = rc[:, 0] >= 2  # 48 foreground tokens
    for seed in range(5):
        keep, (rows, w) = kmeans_plan(emb, rc, 9, np.random.default_rng(seed), allowed, device="cpu")
        assert len(keep) == 9 == len(set(keep)) and np.array_equal(rows, keep) and allowed[keep].all()
        assert np.allclose(w.sum(axis=1), 1) and np.all(w[:, ~allowed] == 0)
        assert np.all((w[:, allowed] > 0).sum(axis=0) == 1)  # every foreground token in exactly one cluster
        assert all(w[i, keep[i]] > 0 for i in range(9))  # each kept token is a member of its own cluster
    a = kmeans_plan(emb, rc, 9, np.random.default_rng(7), allowed, device="cpu")
    b = kmeans_plan(emb, rc, 9, np.random.default_rng(7), allowed, device="cpu")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1][1], b[1][1])  # same seed, same clusters


def test_kmeans_recovers_separated_groups_and_lowers_the_merge_distortion():
    """Six tight groups scattered across the grid (not aligned with the Hilbert strata): k-means recovers them and
    its merge distortion is below the stratified recipe's on every seed."""
    from medprune.remedy import kmeans_plan, merge_plan, select_stratified

    rng = np.random.default_rng(0)
    rc = grid_rc(8, 8)
    centres = rng.normal(size=(6, 16))
    group = rng.integers(0, 6, 64)
    emb = centres[group] + 0.05 * rng.normal(size=(64, 16))
    pure, better = 0, 0
    for seed in range(10):
        keep, (_, w) = kmeans_plan(emb, rc, 6, np.random.default_rng(seed), device="cpu")
        pure += all(len(set(group[np.flatnonzero(row > 0)])) == 1 for row in w)
        ref_keep = select_stratified(rc, 6, np.random.default_rng(seed))
        _, w_ref = merge_plan(emb, ref_keep)
        better += _distortion(emb, w) < _distortion(emb, w_ref)
    assert pure >= 8 and better == 10


def test_d2_seeding_keeps_an_isolated_token():
    """D^2 sampling picks a lone distinctive token (a small lesion) as a centre with high probability; one random token
    per stratum keeps it about k / n of the time."""
    from medprune.remedy import kmeans_plan, select_stratified

    rng = np.random.default_rng(1)
    rc = grid_rc(8, 8)
    emb = np.tile(rng.normal(size=(1, 16)), (64, 1)) + 0.05 * rng.normal(size=(64, 16))
    emb[27] = rng.normal(size=16) * 3  # the outlier
    kept = sum(27 in kmeans_plan(emb, rc, 6, np.random.default_rng(s), device="cpu")[0] for s in range(40))
    strat = sum(27 in select_stratified(rc, 6, np.random.default_rng(s)) for s in range(40))
    assert kept >= 38 and strat <= 15


def test_kmeans_small_gate_keeps_every_allowed_token_and_tops_up():
    from medprune.remedy import kmeans_plan

    rc = grid_rc(8, 8)
    allowed = np.zeros(64, dtype=bool)
    allowed[[3, 9, 40]] = True
    keep, (rows, w) = kmeans_plan(np.random.default_rng(0).normal(size=(64, 8)), rc, 6, np.random.default_rng(0),
                                  allowed, device="cpu")
    assert len(keep) == 6 and {3, 9, 40} <= set(keep) and np.allclose(w, np.eye(64)[keep])


def test_e43_rows_dry_run(tmp_path):
    from medprune.run_eval import RunSpec, run_many

    bundle, out = _bundle(tmp_path), tmp_path / "results"
    specs = [RunSpec("lingshu7b", "slake_en", m, 0.10) for m in ("gate_jitter_merge_pa", "gate_kmeans_merge_pa_eq")]
    run_many(specs, bundle, out, {"lingshu7b": {"slake": 1.0, "text_only": 0.5}}, dry_run=True)
    rows = {s.method: [json.loads(l) for l in open(out / "runs" / f"{s.run_id('dryrun')}.jsonl")] for s in specs}
    assert all(r["k_keep"] == 2 for r in rows["gate_jitter_merge_pa"])  # keep_count(16, 0.10)
    assert all(r["k_keep"] == fastv_matched_count(16, 2, 28) for r in rows["gate_kmeans_merge_pa_eq"])



# ---------------------------------------------------------------- E46: SLIC merge clusters, background + size term

def _spread(rc, weights):
    """Mean squared spatial distance of members to their cluster's centroid (cluster compactness)."""
    out = []
    for row in weights:
        members = np.flatnonzero(row > 0)
        out.append(((rc[members] - rc[members].mean(axis=0)) ** 2).sum(axis=1).mean())
    return float(np.mean(out))


def test_slic_plan_partitions_the_gate_into_exactly_k_clusters():
    from medprune.remedy import slic_plan

    rng = np.random.default_rng(3)
    rc, emb = grid_rc(8, 8), rng.normal(size=(64, 12))
    allowed = rc[:, 0] >= 2
    for seed in range(5):
        keep, (rows, w) = slic_plan(emb, rc, 9, np.random.default_rng(seed), allowed, device="cpu")
        assert len(keep) == 9 == len(set(keep)) and np.array_equal(rows, keep) and allowed[keep].all()
        assert np.allclose(w.sum(axis=1), 1) and np.all(w[:, ~allowed] == 0)
        assert np.all((w[:, allowed] > 0).sum(axis=0) == 1) and all(w[i, keep[i]] > 0 for i in range(9))
    a = slic_plan(emb, rc, 9, np.random.default_rng(7), allowed, device="cpu")
    assert np.array_equal(a[0], slic_plan(emb, rc, 9, np.random.default_rng(7), allowed, device="cpu")[0])


def test_slic_is_spatially_compact_where_kmeans_is_not_and_purer_than_strata():
    """Two content classes in vertical stripes: k-means groups by content across the whole image (wide clusters), the
    stratified merge mixes classes inside its regions; SLIC keeps clusters compact and at least as pure in content."""
    from medprune.remedy import kmeans_plan, merge_plan, select_stratified, slic_plan

    rng = np.random.default_rng(0)
    rc = grid_rc(12, 12)
    cls = ((rc[:, 1] // 3) % 2).astype(int)
    emb = np.array([[4.0, 0], [0, 4.0]])[cls] + 0.3 * rng.normal(size=(144, 2))
    emb = np.hstack([emb, 0.3 * rng.normal(size=(144, 6))])
    spread_s, spread_k, pure_s, pure_r = [], [], [], []
    purity = lambda w: np.mean([np.bincount(cls[w[i] > 0]).max() / (w[i] > 0).sum() for i in range(len(w))])
    for seed in range(8):
        _, (_, w_s) = slic_plan(emb, rc, 12, np.random.default_rng(seed), device="cpu")
        _, (_, w_k) = kmeans_plan(emb, rc, 12, np.random.default_rng(seed), device="cpu")
        _, w_r = merge_plan(emb, select_stratified(rc, 12, np.random.default_rng(seed)))
        spread_s.append(_spread(rc, w_s)); spread_k.append(_spread(rc, w_k))
        pure_s.append(purity(w_s)); pure_r.append(purity(w_r))
    assert np.mean(spread_s) < 0.5 * np.mean(spread_k)
    assert np.mean(pure_s) >= np.mean(pure_r)


def test_slic_small_gate_keeps_every_allowed_token():
    from medprune.remedy import slic_plan

    rc = grid_rc(8, 8)
    allowed = np.zeros(64, dtype=bool)
    allowed[[3, 9, 40]] = True
    keep, (_, w) = slic_plan(np.random.default_rng(0).normal(size=(64, 8)), rc, 6, np.random.default_rng(0), allowed,
                             device="cpu")
    assert len(keep) == 6 and {3, 9, 40} <= set(keep) and np.allclose(w, np.eye(64)[keep])


def test_background_summary_gets_the_size_of_the_whole_background():
    """BG-PA: with the size term, the summary row stands for every background token."""
    from medprune.data import Sample
    from medprune.run_eval import FakeAdapter, RunSpec, build_state
    from medprune.selection import METHODS

    adapter = FakeAdapter("lingshu7b")
    sample = Sample(qid="toy/0", dataset="slake_en", image_id="toy/0", image_path="x.png", question="Is it?",
                    answer="yes", answer_type="yesno", prompt_style="close")
    inputs = adapter.prepare(sample, image=object())
    gate = np.ones((4, 4), dtype=np.float32)
    gate[0, :] = 0.0
    state = build_state(RunSpec("lingshu7b", "slake_en", "gate_jitter_bgsum_pa_eq", 0.25),
                        METHODS["gate_jitter_bgsum_pa_eq"], sample, inputs, gate, None, adapter, None)
    state.size_term_from_merge()
    rows, log_sizes = state.key_bias
    sizes = dict(zip(rows.tolist(), np.round(np.exp(log_sizes)).astype(int).tolist()))
    bg_kept = [i for i in rows if i < 4]
    assert len(bg_kept) == 1 and sizes[bg_kept[0]] == 4 and sum(sizes.values()) == 16


def test_e46_rows_dry_run(tmp_path):
    from medprune.run_eval import RunSpec, run_many

    bundle, out = _bundle(tmp_path), tmp_path / "results"
    specs = [RunSpec("lingshu7b", "slake_en", m, 0.10) for m in ("gate_jitter_bgsum_pa_eq", "gate_slic_merge_pa_eq")]
    run_many(specs, bundle, out, {"lingshu7b": {"slake": 1.0, "text_only": 0.5}}, dry_run=True)
    for s in specs:
        rows = [json.loads(l) for l in open(out / "runs" / f"{s.run_id('dryrun')}.jsonl")]
        assert all(r["k_keep"] == fastv_matched_count(16, 2, 28) for r in rows), s.method


# ------------------------------------------------------------------------------------------------ E47 (10% push)

def test_flat_sched_counts_match_the_flat_budget_from_below():
    """E47: the schedule's mean visual tokens per layer never exceeds the flat count k and is within one stage-2 token
    of it; k1 >= k >= k2 >= 1."""
    from medprune.remedy import flat_sched_counts
    from medprune.selection import keep_count

    for layers, cut in ((28, 20), (36, 26)):
        for n in (16, 36, 49, 64, 81, 100, 121, 256, 324, 1024, 1296):
            k = keep_count(n, 0.10)
            k1, k2 = flat_sched_counts(k, n, layers, cut)
            mean = (cut * k1 + (layers - cut) * k2) / layers
            assert 1 <= k2 <= k1 <= n and k1 >= k
            assert mean <= k + 1e-9 or k2 == 1
            assert k - mean <= (layers - cut) / layers + 1e-9
    assert flat_sched_counts(32, 324, 28, 20) == (40, 12)  # Lingshu SLAKE CT slice: mean 32.0 = k
    assert flat_sched_counts(6, 64, 28, 20) == (7, 3)  # 64-token image: mean 5.86 <= 6


def test_e47_rows_dry_run(tmp_path):
    from medprune.remedy import flat_sched_counts
    from medprune.run_eval import RunSpec, run_many
    from medprune.selection import keep_count

    bundle, out = _bundle(tmp_path), tmp_path / "results"
    names = ("gate_jitter_merge_sched10_pa", "gate_slic_merge_pa", "gate_slic_merge_sched10_pa",
             "gate_jitter_sink_merge_pa", "gate_jitter_bgnull_pa")
    specs = [RunSpec("lingshu7b", "slake_en", m, 0.25) for m in names]
    run_many(specs, bundle, out, {"lingshu7b": {"slake": 1.0, "text_only": 0.5}}, dry_run=True)
    rows = {s.method: [json.loads(l) for l in open(out / "runs" / f"{s.run_id('dryrun')}.jsonl")] for s in specs}
    k = keep_count(16, 0.25)
    k1, k2 = flat_sched_counts(k, 16, 28, 20)
    for name in ("gate_jitter_merge_sched10_pa", "gate_slic_merge_sched10_pa"):
        for r in rows[name]:
            (l1, s1), (l2, s2) = r["keep_stages"]
            assert (l1, l2) == (0, 20) and len(s1) == k1 and len(s2) == k2 and set(s2) < set(s1)
    for name in ("gate_slic_merge_pa", "gate_jitter_sink_merge_pa", "gate_jitter_bgnull_pa"):
        assert all(r["k_keep"] == k and r["keep_stages"] is None for r in rows[name])
    assert all(0 in r["keep_idx"] for r in rows["gate_jitter_sink_merge_pa"])


def _toy_state(name: str, budget: float = 0.25):
    from medprune.data import Sample
    from medprune.run_eval import FakeAdapter, RunSpec, build_state
    from medprune.selection import METHODS

    adapter = FakeAdapter("lingshu7b")
    sample = Sample(qid="toy/1", dataset="slake_en", image_id="toy/1", image_path="x.png", question="Is it?",
                    answer="yes", answer_type="yesno", prompt_style="close")
    inputs = adapter.prepare(sample, image=object())
    gate = np.ones((4, 4), dtype=np.float32)
    gate[0, :] = 0.0  # the top row (tokens 0-3) is background
    return build_state(RunSpec("lingshu7b", "slake_en", name, budget), METHODS[name], sample, inputs, gate, None,
                       adapter, None)


def test_sink_row_keeps_token_zero_as_itself():
    """BG-S: token 0 (background here) is kept unmerged with s = 1; the other k - 1 tokens are foreground strata and
    every dropped foreground token merges into a foreground token; dropped background is discarded."""
    state = _toy_state("gate_jitter_sink_merge_pa")
    rows, w = state.merge
    assert 0 in state.keep and len(state.keep) == 4
    row0 = w[np.searchsorted(rows, 0)]
    assert row0[0] == 1.0 and np.count_nonzero(row0) == 1
    others = [i for i in state.keep if i != 0]
    assert all(i >= 4 for i in others)  # foreground strata only
    assert np.all(w[:, 1:4] == 0)  # background tokens 1-3 discarded
    state.size_term_from_merge()
    assert float(state.key_bias[1][np.searchsorted(rows, 0)]) == 0.0  # log 1


def test_bg_null_row_zeroes_the_summary_value_and_keeps_its_mass():
    """BG-N: one kept background token holds the mean of the dropped background, carries log of its size, and is the
    only zeroed value; without the background the row is the plain recipe."""
    state = _toy_state("gate_jitter_bgnull_pa")
    rows, w = state.merge
    assert state.null_value is not None and state.null_value.size == 1
    slot = int(state.null_value[0])
    assert slot in range(4) and slot in state.keep  # a background token
    assert np.count_nonzero(w[np.searchsorted(rows, slot)]) == 4  # itself + the three dropped background tokens
    state.size_term_from_merge()
    np.testing.assert_allclose(state.key_bias[1][np.searchsorted(rows, slot)], np.log(4), rtol=1e-6)
