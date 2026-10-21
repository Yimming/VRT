"""
Regression tests for the fixes of the code release (run from the repository root: python -m pytest tests -q).

They cover the four problems found in the audit of the as-run scripts: non-finite inputs or statistics turning into a
rejection, resume by row count, four-decimal p-values, and the training resume index. They also pin the behaviour of
the VRT scoring that the paper relies on (reproducible p for a seed, +inf for a float32 duplicate query, no NaN).
"""
import csv
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import (validate_features, check_statistics, ResumeIndex, open_append, exact_perm_p, perm_p,
                           resume_start_epoch)  # noqa: E402
from vrt import compute_chat, vrt_full, vrt_low, check_scores, l2n  # noqa: E402
from baselines import mmd_test_fast  # noqa: E402
from baselines import energy_test_fast  # noqa: E402
from baselines import c2st_test  # noqa: E402


def sphere(n, d, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)) + shift
    return l2n(x)


# ----------------------------------------------------------------------------- 1. finiteness
def test_validate_features_rejects_nan_inf_and_zero_rows():
    good = sphere(50, 8, 0)
    assert validate_features(good) is good
    bad = good.copy(); bad[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        validate_features(bad)
    bad = good.copy(); bad[0, 0] = np.inf
    with pytest.raises(ValueError):
        validate_features(bad)
    zero = good.copy(); zero[5] = 0.0
    with pytest.raises(ValueError, match="zero norm"):
        validate_features(zero)
    validate_features(zero, allow_zero_rows=True)


def test_check_statistics_rejects_non_finite():
    check_statistics(0.1, np.array([0.0, 0.2]))
    with pytest.raises(FloatingPointError):
        check_statistics(np.nan, np.array([0.0, 0.2]))
    with pytest.raises(FloatingPointError):
        check_statistics(0.1, np.array([0.0, np.inf]))


def test_mmd_and_energy_raise_on_nan_instead_of_rejecting():
    tr, te = sphere(120, 6, 1), sphere(100, 6, 2)
    obs, p = mmd_test_fast(tr, te, n_sub=60, n_perm=50, seed=42, backend="numpy")
    assert np.isfinite(obs) and 0 < p <= 1
    bad = te.copy(); bad[:, 0] = np.nan                     # every row, so that any subsample contains it
    with pytest.raises(ValueError):
        mmd_test_fast(tr, bad, n_sub=60, n_perm=50, seed=42, backend="numpy")
    with pytest.raises(ValueError):
        energy_test_fast(tr, bad, n_sub=60, n_perm=50, seed=42, backend="numpy")
    with pytest.raises(ValueError):
        c2st_test(tr, bad, n_sub=60, seed=42)


# ----------------------------------------------------------------------------- 2. resume by key
def test_resume_index_keys_and_conflicts(tmp_path):
    path = tmp_path / "out.csv"
    f, w = open_append(str(path), ["tag", "epoch", "seed", "p"])
    w.writerow(["m", 0, 42, "0.5"]); w.writerow(["m", 0, 43, "0.4"]); w.writerow(["m", 1, 42, "0.3"])
    f.close()
    idx = ResumeIndex(str(path), key_cols=("epoch", "seed"), value_cols=("p",))
    assert idx.has(0, 42) and idx.has(0, 43) and idx.has(1, 42) and not idx.has(1, 43)
    assert idx.epochs_complete([42, 43]) == {0}
    assert "3 keys" in idx.report()
    # a second writer appended the same key with the same value: ignored; with another value: refused
    f, w = open_append(str(path), ["tag", "epoch", "seed", "p"])
    w.writerow(["m", 0, 42, "0.5"]); w.writerow(["m", 1, 42, "0.9"]); f.close()
    idx = ResumeIndex(str(path), key_cols=("epoch", "seed"), value_cols=("p",))
    assert idx.duplicates == 2 and idx.conflicts == [(1, 42)]
    with pytest.raises(RuntimeError, match="conflicting"):
        idx.report()
    # a torn line is not a completed key
    with open(path, "a") as fh:
        fh.write("m,2\n")                                     # cut after the epoch field
    idx = ResumeIndex(str(path), key_cols=("epoch", "seed"))
    assert not any(k[0] == 2 for k in idx.rows)


def test_open_append_writes_header_once(tmp_path):
    path = str(tmp_path / "a.csv")
    f, w = open_append(path, ["a", "b"]); w.writerow([1, 2]); f.close()
    f, w = open_append(path, ["a", "b"]); w.writerow([3, 4]); f.close()
    rows = list(csv.reader(open(path)))
    assert rows == [["a", "b"], ["1", "2"], ["3", "4"]]


# ----------------------------------------------------------------------------- 3. p-values
def test_exact_perm_p_recovers_every_lattice_value():
    n_perm = 2000
    for k in range(1, n_perm + 2):
        p = k / (n_perm + 1)
        assert exact_perm_p(round(p, 4), n_perm) == p
        assert exact_perm_p(float(f"{p:.6f}"), n_perm) == p
    assert perm_p(99, n_perm) == 100 / 2001


def test_appendix_g_rounding_case():
    """the two central stored values 0.0470 and 0.0530 of zoo_c10_m3 epoch 30 average to 0.0500 and fail "< 0.05";
    the exact values 94/2001 and 106/2001 average to 100/2001 = 0.049975 and reject"""
    stored = [0.0470, 0.0530]
    assert not (np.median(stored) < 0.05)
    exact = [exact_perm_p(p, 2000) for p in stored]
    assert exact == [94 / 2001, 106 / 2001]
    assert np.median(exact) == 100 / 2001 < 0.05


# ----------------------------------------------------------------------------- 4. training resume
def test_resume_start_epoch(tmp_path):
    assert resume_start_epoch(str(tmp_path), 200) == (0, None)
    open(tmp_path / "checkpoint_000.pt", "w").close()
    assert resume_start_epoch(str(tmp_path), 200) == (0, str(tmp_path / "checkpoint_000.pt"))
    open(tmp_path / "checkpoint_060.pt", "w").close()
    start, path = resume_start_epoch(str(tmp_path), 200)
    assert start == 60 and path.endswith("checkpoint_060.pt")     # the next checkpoint written is checkpoint_061


# ----------------------------------------------------------------------------- 5. VRT scoring behaviour
def test_vrt_full_is_reproducible_and_finite():
    tr, te = sphere(400, 16, 3), sphere(300, 16, 4, shift=0.05)
    a = vrt_full(tr, te, 16, k=10, M=100, n_qtr=200, n_perm=200, seed=42)
    b = vrt_full(tr, te, 16, k=10, M=100, n_qtr=200, n_perm=200, seed=42)
    assert a == b and np.isfinite(a[0]) and 0 < a[1] <= 1
    c = vrt_low(tr.astype(np.float64), te.astype(np.float64), 16, k=5, M=50, n_qtr=100, n_qte=150, n_perm=200, seed=42)
    assert np.isfinite(c[0]) and 0 < c[1] <= 1


def test_float32_duplicate_query_gives_posinf_and_is_tolerated_by_the_median_test():
    """documents the as-run behaviour: a query equal to a reference point has c-hat = +inf in float32 (the 1e-300 floor
    is cast to 0) and a large finite value in float64; the median test accepts +inf, check_scores refuses NaN"""
    ref = sphere(60, 32, 5)
    q = np.vstack([ref[:3], sphere(20, 32, 6)])
    c32 = compute_chat(q.astype(np.float32), ref.astype(np.float32), 32, 8)
    c64 = compute_chat(q.astype(np.float64), ref.astype(np.float64), 32, 8)
    assert np.isposinf(c32[:3]).all() and np.isfinite(c32[3:]).all()
    assert np.isfinite(c64).all() and (c64[:3] > c64[3:].max()).all()
    check_scores(c32)                                   # +inf allowed
    with pytest.raises(FloatingPointError):
        check_scores(np.array([1.0, np.nan]))
    with pytest.raises(FloatingPointError):
        check_scores(np.array([1.0, -np.inf]))
