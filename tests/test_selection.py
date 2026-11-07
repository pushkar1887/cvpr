import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from medprune.selection import (METHODS, keep_count, pdrop_schedule, rng_for, sample_map, select_divprune,
                                select_inverse, select_random, select_static, select_topk, select_uniform)


def assert_valid(idx: np.ndarray, n: int, k: int) -> None:
    assert len(idx) == k and len(set(idx.tolist())) == k
    assert np.all(np.diff(idx) > 0) and idx.min() >= 0 and idx.max() < n


def test_keep_count_matches_plan_token_counts():
    assert keep_count(324, 0.10) == 32  # Lingshu, 512x512
    assert keep_count(256, 0.10) == 26  # Qwen3-VL, 512x512
    assert keep_count(576, 0.10) == 58  # HuatuoGPT
    assert keep_count(64, 0.10) == 6  # 224x224 on Qwen models
    assert keep_count(5, 0.01) == 1
    assert keep_count(10, 1.0) == 10


def test_random_depends_only_on_seed_and_question():
    a = select_random(324, 32, rng_for(0, "slake_en/1"))
    np.testing.assert_array_equal(a, select_random(324, 32, rng_for(0, "slake_en/1")))
    assert not np.array_equal(a, select_random(324, 32, rng_for(1, "slake_en/1")))
    assert not np.array_equal(a, select_random(324, 32, rng_for(0, "slake_en/2")))
    assert_valid(a, 324, 32)


def test_random_is_identical_across_processes():
    code = "from medprune.selection import rng_for; print(rng_for(0, 'q').integers(0, 10**9))"
    repo = str(Path(__file__).resolve().parent.parent)
    outputs = set()
    for hash_seed in ("1", "2"):  # different str-hash salts must not change the keep-set
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": repo}
        outputs.add(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                                   env=env).stdout.strip())
    assert outputs == {str(rng_for(0, "q").integers(0, 10**9))}


def test_uniform_spacing_and_restriction():
    idx = select_uniform(100, 10)
    assert_valid(idx, 100, 10)
    assert idx[0] == 0 and idx[-1] == 99
    allowed = np.zeros(100, bool)
    allowed[50:] = True
    assert np.all(select_uniform(100, 10, allowed) >= 50)
    allowed = np.zeros(100, bool)
    allowed[:4] = True
    idx = select_uniform(100, 10, allowed)
    assert_valid(idx, 100, 10)
    assert set(range(4)) <= set(idx.tolist())


def test_topk_respects_gate_then_fills_by_score():
    scores = np.arange(20, dtype=float)
    assert select_topk(scores, 5).tolist() == [15, 16, 17, 18, 19]
    allowed = np.zeros(20, bool)
    allowed[:10] = True
    assert select_topk(scores, 5, allowed).tolist() == [5, 6, 7, 8, 9]
    allowed = np.zeros(20, bool)
    allowed[[0, 1]] = True
    assert select_topk(scores, 5, allowed).tolist() == [0, 1, 17, 18, 19]


def test_inverse_gate_takes_background_first():
    allowed = np.ones(50, bool)
    allowed[:10] = False
    idx = select_inverse(allowed, 5)
    assert_valid(idx, 50, 5)
    assert np.all(idx < 10)
    idx = select_inverse(allowed, 15)
    assert_valid(idx, 50, 15)
    assert set(range(10)) <= set(idx.tolist())


def test_divprune_covers_distinct_clusters():
    rng = np.random.default_rng(0)
    feats = np.repeat(np.eye(4) * 10, 5, axis=0) + rng.normal(scale=0.01, size=(20, 4))
    idx = select_divprune(feats, 4)
    assert_valid(idx, 20, 4)
    assert sorted(set((idx // 5).tolist())) == [0, 1, 2, 3]
    allowed = np.zeros(20, bool)
    allowed[:10] = True
    idx = select_divprune(feats, 2, allowed)
    assert np.all(idx < 10) and len(set((idx // 5).tolist())) == 2


def test_static_map_picks_the_mapped_region():
    m = np.zeros((32, 32))
    m[16:, 16:] = 1.0  # bottom-right quadrant
    s = sample_map(m, (4, 4)).reshape(4, 4)
    assert s[3, 3] == 1.0 and s[0, 0] == 0.0
    assert select_static(m, (4, 4), 4).tolist() == [10, 11, 14, 15]


def test_pdrop_is_token_matched_to_fastv():
    schedule = pdrop_schedule(0.10, 28)
    assert [b for b, _ in schedule] == [2, 9, 16, 23]
    assert schedule[0][1] == pytest.approx(0.2019, abs=1e-4)
    fractions = [1.0] * 2 + [schedule[0][1]] * 7 + [schedule[1][1]] * 7 + [schedule[2][1]] * 7 + [schedule[3][1]] * 5
    assert np.mean(fractions) == pytest.approx((2 + 26 * 0.10) / 28)


def test_registry_is_consistent_with_fastv_definition():
    assert len(METHODS) == 52  # + sparsevlm, vispruner, prunesid, btp (E13); 2 rater ablations (E16); 8 pruners (E14);
    #                            FastV K = 3-6 (E17)
    for spec in METHODS.values():
        if spec.rule == "topk" and spec.score is not None:
            assert spec.mask_from == spec.score_layer + 1 or spec.two_pass, spec.name
        assert spec.merge in (None, "off", "nearest"), spec.name
    assert METHODS["fastv"].score_layer == 1 and METHODS["fastv"].mask_from == 2
    assert METHODS["fastv_bottom"].invert and not METHODS["fastv"].invert
    assert METHODS["gate_cdpruner_neg"].invert and METHODS["gate_cdpruner_neg"].gate == "foreground"
    assert METHODS["fastv_text"].score == "text" and METHODS["fastv_text"].mask_from == 2
    assert METHODS["full_profile"].profile and METHODS["full_profile"].rule == "none"


def test_gated_random_stays_inside_the_gate_and_tops_up():
    rng = np.random.default_rng(0)
    allowed = np.zeros(64, bool)
    allowed[10:30] = True
    keep = select_random(64, 6, rng, allowed)
    assert keep.size == 6 and allowed[keep].all()
    small = np.zeros(64, bool)
    small[:3] = True
    keep = select_random(64, 6, np.random.default_rng(1), small)
    assert keep.size == 6 and set(range(3)) <= set(keep.tolist())


def test_ungated_random_draw_is_unchanged():
    a = select_random(64, 6, np.random.default_rng(7))
    b = np.sort(np.random.default_rng(7).choice(64, size=6, replace=False))
    assert np.array_equal(a, b)
