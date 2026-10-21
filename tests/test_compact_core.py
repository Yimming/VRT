"""Regression coverage for consolidated numerical routines."""
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baselines
import vrt

# Recorded from the independently audited release before consolidation.
VRT_EXPECTED = {'float32_0_full': [-0.017062614361445072, 0.7419354838709677],
 'float32_0_reversed': [0.05549665292104078, 0.45161290322580644],
 'float32_0_small': [-0.004672775665918949, 0.967741935483871],
 'float32_42_full': [-0.044590552647908455, 0.7580645161290323],
 'float32_42_reversed': [0.18622866272926353, 0.016129032258064516],
 'float32_42_small': [-0.05772358179092407, 0.4838709677419355],
 'float32_91_full': [-0.19262074430783604, 0.016129032258064516],
 'float32_91_reversed': [-0.06090326110521971, 0.24193548387096775],
 'float32_91_small': [-0.19262074430783604, 0.016129032258064516],
 'float64_0_full': [-0.01706309249815363, 0.7419354838709677],
 'float64_0_reversed': [0.0554977890230548, 0.45161290322580644],
 'float64_0_small': [-0.004673294665240135, 0.967741935483871],
 'float64_42_full': [-0.044591607533649213, 0.7580645161290323],
 'float64_42_reversed': [0.18622952967524964, 0.016129032258064516],
 'float64_42_small': [-0.05772458099936939, 0.4838709677419355],
 'float64_91_full': [-0.19262167563472, 0.016129032258064516],
 'float64_91_reversed': [-0.06090343526501196, 0.24193548387096775],
 'float64_91_small': [-0.19262167563472, 0.016129032258064516]}

@pytest.mark.parametrize("case", sorted(VRT_EXPECTED))
def test_vrt_preserves_protocol_results(case):
    dtype_name, seed, mode = case.split("_")
    dtype = getattr(np, dtype_name)
    rng = np.random.RandomState(73)
    tr = vrt.l2n(rng.normal(size=(180, 16)).astype(dtype))
    te = vrt.l2n((rng.normal(size=(130, 16)) + .15).astype(dtype))
    options = dict(k=7, M=45, n_perm=61, seed=int(seed))
    with threadpool_limits(limits=1):
        if mode == "full":
            actual = vrt.vrt_full(tr, te, 16, n_qtr=80, **options)
        elif mode == "small":
            actual = vrt.vrt_low(tr, te, 16, n_qtr=80, n_qte=100, **options)
        else:
            actual = vrt.vrt_reversed(tr, te, 16, n_q=80, **options)
    expected = VRT_EXPECTED[case]
    tolerance = 3e-6 if dtype == np.float32 else 1e-12
    assert actual[0] == pytest.approx(expected[0], abs=tolerance)
    assert actual[1] == expected[1]


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_cached_baselines_match_direct_recomputation(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    rng = np.random.RandomState(7)
    tr = rng.normal(size=(45, 8))
    te = rng.normal(size=(32, 8)) + .2
    options = dict(n_sub=50, n_perm=41, seed=43)
    with threadpool_limits(limits=1):
        for fast, direct in [
            (baselines.mmd_test_fast, baselines.mmd_test_original),
            (baselines.energy_test_fast, baselines.energy_distance_test_original),
        ]:
            expected = direct(tr, te, **options)
            actual = fast(tr, te, backend=backend, **options)
            assert actual[0] == pytest.approx(expected[0], abs=1e-10)
            assert actual[1] == expected[1]


@pytest.mark.parametrize("backend", ["sklearn", "torch"])
def test_knn_graph_and_conditional_moments(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    points = np.random.RandomState(8).normal(size=(8, 4))
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    expected = np.argsort(distances, axis=1)[:, :2]
    idx = baselines.knn_graph(points, 2, backend)
    assert np.array_equal(np.sort(idx, axis=1), np.sort(expected, axis=1))
    counts = []
    for group in combinations(range(8), 3):
        labels = np.zeros(8, dtype=int)
        labels[list(group)] = 1
        counts.append(np.sum(labels[:, None] == labels[idx]))
    mean, variance = baselines.knn_conditional_moments(idx, 3, 5, 2)
    assert mean == pytest.approx(np.mean(counts), abs=1e-12)
    assert variance == pytest.approx(np.var(counts), abs=1e-12)
