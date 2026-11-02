import numpy as np
import pytest

from medprune.stats import (background_retention_ratio, cluster_bootstrap_ci, jaccard, keep_map, mcnemar_exact,
                            paired_delta_ci, quadrant_concentration, static_map_from_scores)


def test_cluster_bootstrap_is_reproducible_and_wider_when_questions_cluster():
    rng = np.random.default_rng(0)
    image_effect = rng.normal(0, 1, 40)
    clusters = np.repeat(np.arange(40), 25)
    values = (image_effect[clusters] + rng.normal(0, 0.1, clusters.size) > 0).astype(float)
    a = cluster_bootstrap_ci(values, clusters, B=500, seed=1)
    assert a == cluster_bootstrap_ci(values, clusters, B=500, seed=1)
    naive = cluster_bootstrap_ci(values, np.arange(values.size), B=500, seed=1)
    assert (a[2] - a[1]) > 2 * (naive[2] - naive[1])


def test_paired_delta_sign():
    a, b = np.ones(100), np.zeros(100)
    mean, low, high = paired_delta_ci(a, b, np.arange(100) // 10, B=200)
    assert mean == 1.0 and low == 1.0 and high == 1.0


def test_mcnemar():
    a = np.array([1] * 30 + [0] * 70, bool)
    assert mcnemar_exact(a, a) == 1.0
    assert mcnemar_exact(a, ~a) == pytest.approx(7.85e-5, rel=1e-2)  # 30 vs 70 discordant pairs, exact binomial
    assert mcnemar_exact(a, ~a) == mcnemar_exact(~a, a)


def test_random_keep_sets_have_retention_ratio_one():
    rng = np.random.default_rng(0)
    background = np.zeros(324, bool)
    background[:120] = True
    ratios = [background_retention_ratio(rng.choice(324, 32, replace=False), background) for _ in range(4000)]
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.02)
    assert np.isnan(background_retention_ratio(np.arange(5), np.zeros(10, bool)))


def test_keep_map_and_quadrant_concentration():
    grids = [(4, 4)] * 10
    bottom_right = [np.array([10, 11, 14, 15])] * 10
    kmap = keep_map(bottom_right, grids, res=8)
    assert kmap.shape == (8, 8) and kmap[7, 7] == 1.0 and kmap[0, 0] == 0.0
    assert quadrant_concentration(kmap, "bottom_right") == pytest.approx(4.0)
    assert quadrant_concentration(np.ones((8, 8))) == pytest.approx(1.0)


def test_static_map_from_scores():
    scores = [np.arange(16, dtype=float)] * 3  # raster-increasing: highest at the bottom-right
    m = static_map_from_scores(scores, [(4, 4)] * 3, res=8)
    assert m.shape == (8, 8) and 0.0 <= m.min() and m.max() <= 1.0 and m[7, 7] > m[0, 0]


def test_jaccard():
    assert jaccard(np.array([1, 2, 3]), np.array([2, 3, 4])) == 0.5
