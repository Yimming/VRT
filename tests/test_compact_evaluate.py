"""Preprocessing and command-level regression tests."""
import csv
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate
from common import l2n
from vrt import perm_p


def test_published_feature_dtypes_and_raw_zero_vectors():
    x = np.ones((1500, 4), dtype=np.float64)
    for method in ("c2st", "mmdagg"):
        raw, _ = evaluate.prepare_features(x * 0, x, method, 1500, "supervised", raw=True)
        assert raw.dtype == np.float32
        assert np.count_nonzero(raw) == 0
    raw, _ = evaluate.prepare_features(x, x, "mmd", 1500, "supervised", raw=True)
    assert raw.dtype == np.float64
    with pytest.raises(ValueError, match="require normalized"):
        evaluate.prepare_features(x, x, "vrt", 1500, "supervised", raw=True)
    main = np.ones((10000, 4))
    for protocol, budget, expected in [("simclr", 10000, np.float32),
                                        ("simclr", 1500, np.float64),
                                        ("supervised", 10000, np.float64)]:
        actual, _ = evaluate.prepare_features(main, main, "vrt", budget, protocol)
        assert actual.dtype == expected


def test_score_statistics_preserve_vrt_permutations():
    rng = np.random.RandomState(19)
    a, b, stream = evaluate.score_samples(l2n(rng.normal(size=(200, 12))),
        l2n(rng.normal(size=(110, 12))), 42, reference=50, neighbours=4, queries=100)
    second_stream = np.random.RandomState()
    second_stream.set_state(stream.get_state())
    statistic, p = perm_p(a, b, 31, second_stream)
    actual, all_p = evaluate.score_statistics_test(a, b, stream, 31)
    assert actual[0] == statistic
    assert all_p[0] == p
    with pytest.raises(ValueError, match="finite"):
        evaluate.score_statistics_test(np.array([np.inf]), b, stream, 31)


def test_small_vrt_command_creates_seed_records(tmp_path):
    rng = np.random.RandomState(41)
    features = tmp_path / "ep000.npz"
    np.savez(features, h_tr=rng.normal(size=(1700, 8)), h_te=rng.normal(size=(1500, 8)))
    output = tmp_path / "test.csv"
    args = ["test", "--features", str(features), "--model", "example", "--epochs", "0",
            "--budget", "1500", "--seeds", "42-43", "--permutations", "19", "--out", str(output)]
    evaluate.main(args)
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [r["seed"] for r in rows] == ["42", "43"]
    assert all(r["model"] == "example" and r["method"] == "vrt" for r in rows)
    assert all(float(r["p"]) >= 1 / 20 for r in rows)
    with pytest.raises(FileExistsError):
        evaluate.main(args)


def test_geometry_command_requires_aligned_labels(tmp_path):
    rng = np.random.RandomState(2)
    x, y = rng.normal(size=(100, 5)), rng.normal(size=(40, 5))
    np.savez(tmp_path / "features.npz", h_tr=x, h_te=y)
    np.savez(tmp_path / "labels.npz", y_tr=np.repeat([0, 1], 50), y_te=np.repeat([0, 1], 20))
    output = tmp_path / "geometry.csv"
    args = ["geometry", "--features", str(tmp_path / "features.npz"),
            "--labels", str(tmp_path / "labels.npz"), "--model", "example", "--epochs", "0",
            "--out", str(output)]
    evaluate.main(args)
    with output.open() as handle:
        row = next(csv.DictReader(handle))
    assert 0 <= float(row["theta_bar_deg"]) <= 180
    assert -1 <= float(row["dR_bar"]) <= 1
    np.savez(tmp_path / "labels.npz", y_tr=np.repeat([0, 1], 40), y_te=np.repeat([0, 1], 20))
    args[-1] = str(tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="Label arrays must match"):
        evaluate.main(args)
