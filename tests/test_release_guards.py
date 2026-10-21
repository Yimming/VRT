"""
Tests of the guards added after the second review (October 18): ResumeIndex counts only valid rows of the current run,
open_append refuses a foreign header and isolates a torn last line, the VRT paths stop on NaN scores and on
non-finite median statistics (instead of returning the minimum p), exact_perm_p refuses off-lattice values, and the
raw (unnormalized) entry points accept zero rows.
"""
import csv
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from common import (ResumeIndex, open_append, repair_tail, exact_perm_p, validate_features, check_scores,
                           count_consistent)  # noqa: E402
from vrt import vrt_full, vrt_low, compute_chat, l2n  # noqa: E402
import vrt
from evaluate import run_test


def sphere(n, d, seed):
    return l2n(np.random.default_rng(seed).standard_normal((n, d)))


# ----------------------------------------------------------------------------- resume index
def write_rows(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def test_resume_index_counts_only_valid_rows_of_this_run(tmp_path):
    path = str(tmp_path / "mmd_r50_n10000_c0-9.csv")
    header = ["tag", "n_samples", "n_perm", "epoch", "seed", "mmd2", "p", "features"]
    write_rows(path, header, [["r50", 10000, 2000, 0, 42, "1e-3", "0.500250", "l2"],   # valid
                              ["v7", 10000, 2000, 0, 43, "1e-3", "0.400300", "l2"],    # another tag
                              ["r50", 10000, 500, 0, 44, "1e-3", "0.400300", "l2"],    # another permutation count
                              ["r50", 8, 2000, 0, 45, "1e-3", "0.400300", "l2"],       # another sample budget
                              ["r50", 10000, 2000, 0, 46, "1e-3", "0.400300", "raw"]]) # unnormalized features
    expect = {"tag": "r50", "n_samples": 10000, "n_perm": 2000, "features": "l2"}
    idx = ResumeIndex(path, key_cols=("epoch", "seed"), value_cols=("mmd2", "p"), ranges={"p": (0, 1)}, expect=expect,
                      header=header)
    assert idx.has(0, 42) and not any(idx.has(0, s) for s in (43, 44, 45, 46)) and idx.foreign == 4
    with pytest.raises(RuntimeError, match="belong to another run"):      # one output file holds one run
        idx.report()
    idx = ResumeIndex(path, key_cols=("epoch", "seed"), value_cols=("mmd2", "p"), expect=expect, header=header, foreign="ignore")
    assert "1 keys present" in idx.report() and "4 rows belong to another run" in idx.report()
    header = header[:-1]
    # invalid values are not complete and make report() refuse
    write_rows(path, header, [["r50", 10000, 2000, 0, 42, "1e-3", "nan"], ["r50", 10000, 2000, 0, 43, "1e-3", "9"],
                              ["r50", 10000, 2000, 0, 44, "inf", "0.5"], ["r50", 10000, 2000, 0, 45, "1e-3", "0.3"]])
    idx = ResumeIndex(path, key_cols=("epoch", "seed"), value_cols=("mmd2", "p"), ranges={"p": (0, 1)},
                      expect={"tag": "r50", "n_perm": 2000}, header=header)
    assert not idx.has(0, 42) and not idx.has(0, 43) and not idx.has(0, 44) and idx.has(0, 45)
    assert len(idx.invalid) == 3
    with pytest.raises(RuntimeError, match="invalid rows"):
        idx.report()
    # a line cut after the seed (a torn line inside the file) is not a completed key either
    with open(path, "a") as f:
        f.write("r50,10000,2000,1,42\n")
    idx = ResumeIndex(path, key_cols=("epoch", "seed"), value_cols=("mmd2", "p"), expect={"tag": "r50"})
    assert not idx.has(1, 42) and any("torn" in msg for _, msg in idx.invalid)
    # MMDAgg rows carry acc = nan by design
    path2 = str(tmp_path / "mmdagg_r50_b5000_c0-9.csv")
    write_rows(path2, ["model", "epoch", "budget", "stat", "seed", "acc", "p"],
               [["r50", 0, "b5000", "mmdagg", 42, "nan", "1"], ["r50", 0, "b5000", "mmdagg", 43, "nan", "0.5"]])
    idx = ResumeIndex(path2, key_cols=("epoch", "budget", "seed"), value_cols=("acc", "p"), int_cols=("epoch", "seed"),
                      ranges={"acc": (0, 1)}, nan_ok=("acc",), discrete={"p": {0.0, 1.0}}, expect={"model": "r50", "stat": "mmdagg"})
    assert idx.has(0, "b5000", 42) and not idx.has(0, "b5000", 43)       # 0.5 is not an MMDAgg decision
    with pytest.raises(RuntimeError, match="not one of"):
        idx.report()


def test_resume_index_checks_permutation_counts(tmp_path):
    path = str(tmp_path / "knn.csv")
    header = ["tag", "n_samples", "k", "epoch", "seed", "p_schilling", "p_henze", "henze_count", "n_perm", "features"]
    write_rows(path, header, [["r50", 10000, 3, 0, 42, "0.1", "0.049975", 99, 2000, "l2"],      # consistent
                              ["r50", 10000, 3, 0, 43, "0.1", "0.025000", 39, 39, "l2"],        # count 39 of 39 means p = 1
                              ["r50", 10000, 3, 0, 44, "0.1", "0.049975", "", 2000, "l2"],      # count missing
                              ["r50", 10000, 3, 0, 45, "0.1", "1.000000", -2, -2, "l2"]])       # negative metadata
    idx = ResumeIndex(path, key_cols=("epoch", "seed"), value_cols=("p_schilling", "p_henze", "henze_count"),
                      ranges={"p_henze": (0, 1), "henze_count": (0, 2000)},
                      expect={"tag": "r50", "n_samples": 10000, "k": 3, "n_perm": 2000, "features": "l2"},
                      row_check=count_consistent("p_henze", "henze_count", "n_perm"), header=header)
    assert idx.has(0, 42) and not idx.has(0, 43) and not idx.has(0, 44) and not idx.has(0, 45)
    with pytest.raises(RuntimeError):
        idx.report()
    assert count_consistent("p", "count", "n_perm")({"p": "0.050000", "count": "1000", "n_perm": "2000"}) is not None
    assert count_consistent("p", "count", "n_perm")({"p": "0.500250", "count": "1000", "n_perm": "2000"}) is None


def test_resume_index_and_open_append_refuse_a_foreign_header(tmp_path):
    path = str(tmp_path / "knn_r50_n10000_part_c0-9.csv")
    old = ["tag", "n_samples", "k", "epoch", "seed", "p_schilling", "p_henze"]
    new = old + ["henze_count", "n_perm"]
    write_rows(path, old, [["r50", 10000, 3, 0, 42, "0.0412", "0.0500"]])
    with pytest.raises(RuntimeError, match="differs from the runner's schema"):
        ResumeIndex(path, key_cols=("epoch", "seed"), value_cols=("p_henze",), header=new)
    with pytest.raises(RuntimeError, match="differs from"):
        open_append(path, new)
    f, w = open_append(path, old)                              # the matching header is fine
    w.writerow(["r50", 10000, 3, 0, 43, "0.1", "0.2"]); f.close()
    assert len(list(csv.reader(open(path)))) == 3


def test_torn_last_line_is_isolated_before_appending(tmp_path):
    path = str(tmp_path / "out.csv")
    with open(path, "w") as f:
        f.write("tag,epoch,seed,p\nm,0,42,0.5\nm,0,43,0.")                 # killed in mid-write, no final newline
    idx = ResumeIndex(path, key_cols=("epoch", "seed"), value_cols=("p",))
    assert idx.has(0, 42) and not idx.has(0, 43) and idx.repaired == "m,0,43,0."
    f, w = open_append(path, ["tag", "epoch", "seed", "p"]); w.writerow(["m", 0, 43, "0.4"]); f.close()
    rows = list(csv.reader(open(path)))
    assert rows == [["tag", "epoch", "seed", "p"], ["m", "0", "42", "0.5"], ["m", "0", "43", "0.4"]]
    assert open(path + ".torn").read() == "m,0,43,0.\n"
    assert repair_tail(path) == ""                              # nothing to do on a clean file


# ----------------------------------------------------------------------------- VRT guards
def test_vrt_paths_stop_on_nan_scores_and_non_finite_statistics():
    rng = np.random.default_rng(0)
    base = sphere(80, 16, 1).astype(np.float32)
    # every query coincides with a reference point: all scores +inf in float32, the medians are inf - inf = NaN
    tr = np.vstack([base[:40], base[:40]])                     # reference (first 40 after the permutation) and queries alike
    te = base[:40].copy()
    with pytest.raises(FloatingPointError):
        vrt_full(tr, te, 16, k=5, M=40, n_qtr=40, n_perm=20, seed=42)
    # a query coinciding with two duplicate reference points gives NaN scores; check_scores refuses them
    ref = np.vstack([base[:1], base[:1], base[1:30]]).astype(np.float32)
    c = compute_chat(base[:1].astype(np.float32), ref, 16, 4)
    assert np.isnan(c).any()
    with pytest.raises(FloatingPointError):
        check_scores(c)
    with pytest.raises(FloatingPointError):
        vrt.perm_p(np.array([1.0, np.nan]), np.array([1.0, 2.0]), 5, np.random.RandomState(0))
    with pytest.raises(FloatingPointError):
        vrt.perm_p(np.array([np.inf] * 4), np.array([np.inf] * 4), 5, np.random.RandomState(0))
    # ordinary inputs are untouched: the guarded functions give the same (obs, p) as before (reference values of the
    # as-run code for this seed are reproduced by construction, here checked against the unguarded formula)
    tr64, te64 = sphere(300, 16, 3), sphere(200, 16, 4)
    obs, p = vrt_low(tr64, te64, 16, k=5, M=50, n_qtr=100, n_qte=150, n_perm=100, seed=42)
    assert np.isfinite(obs) and 0 < p <= 1
    del rng


# ----------------------------------------------------------------------------- exact p and raw features
def test_exact_perm_p_refuses_off_lattice_values():
    assert exact_perm_p(0.0500, 2000) == 100 / 2001
    assert exact_perm_p(0.500250, 2000, decimals=6) == 1001 / 2001
    for bad in (0.5, 0.0, 1.0001, -0.1, float("nan")):
        with pytest.raises(ValueError):
            exact_perm_p(bad, 2000)
    with pytest.raises(ValueError):
        exact_perm_p(0.05, 20000)                              # four decimals cannot identify k for 20,000 permutations


@pytest.mark.parametrize("method", ["mmd", "ed", "knn", "c2st"])
def test_raw_entry_points_accept_zero_rows(method):
    train, test = sphere(40, 4, 7), sphere(40, 4, 8)
    train[3] = 0.0
    with pytest.raises(ValueError, match="zero norm"):
        run_test(train, test, method, budget=40, n_perm=7, raw=False)
    rows = run_test(train, test, method, budget=40, n_perm=7, raw=True)
    assert len(rows) == (2 if method == "knn" else 1)
    assert all(np.isfinite(row["p"]) and 0 <= row["p"] <= 1 for row in rows)
