"""Coverage and input checks for the consolidated result commands."""
import csv
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from results import SchemaError, ResultRoutes, aggregate_new_runs, manifest_check, sha256, validate_tree

FIELDS = ["model", "epoch", "method", "budget", "features", "protocol", "seed", "n_perm", "backend",
          "statistic", "p", "accuracy", "decision"]
SEEDS = set(range(42, 47))


def write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def row(method, epoch, seed, p=.01, acc="", decision="", backend="cpu"):
    return dict(model="example", epoch=epoch, method=method, budget=1500, features="l2", protocol="supervised",
                seed=seed, n_perm=2000, backend=backend, statistic="", p=p, accuracy=acc, decision=decision)


def test_new_runs_keep_rules_and_incomplete_checkpoints(tmp_path):
    rows = []
    for epoch in range(4):
        for seed in sorted(SEEDS):
            # A transient VRT alarm at 1 must not move sustained detection before 3.
            rows.append(row("vrt", epoch, seed, p=.01 if epoch in (1, 3) else .5))
            rows.append(row("c2st", epoch, seed, p=.01, acc=.56 if epoch == 3 else .55))
            # Exactly 60% of MMDAgg decisions is enough, starting at 2.
            rows.append(row("mmdagg", epoch, seed, p="", decision=int(epoch >= 2 and seed < 45)))
    rows.append(row("vrt", 4, 42))
    path = tmp_path / "runs.csv"
    write(path, FIELDS, rows)
    result = {r["method"]: r for r in aggregate_new_runs([path], SEEDS)}
    assert result["vrt"]["onset"] == 3
    assert result["c2st"]["onset"] == 3
    assert result["mmdagg"]["onset"] == 2
    assert result["vrt"]["partial_epochs"] == 1
    assert result["vrt"]["missing_seed_epochs"] == "4"
    assert all(r["unevaluated_after_onset"] == 1 for r in result.values())


def test_new_runs_refuse_nan_conflicts_and_wrong_decisions(tmp_path):
    path = tmp_path / "runs.csv"
    for bad in [row("vrt", 0, 42, p="nan"), row("mmdagg", 0, 42, p="", decision=.5),
                row("c2st", 0, 42, p=.01, acc="nan")]:
        write(path, FIELDS, [bad])
        with pytest.raises(SchemaError):
            aggregate_new_runs([path], SEEDS)
    write(path, FIELDS, [row("vrt", 0, 42, p=.01), row("vrt", 0, 42, p=.02)])
    with pytest.raises(SchemaError, match="conflicting values"):
        aggregate_new_runs([path], SEEDS)
    write(path, FIELDS, [row("vrt", 0, s, backend=b) for b in ["cpu", "gpu"] for s in SEEDS])
    assert len(aggregate_new_runs([path], SEEDS)) == 2


def test_route_caches_are_separate_between_data_directories(tmp_path):
    fields = ["tag", "budget", "epoch", "seed", "dc", "p"]
    for root, p in [(tmp_path / "first", .01), (tmp_path / "second", .9)]:
        write(root / "vrt_fwd/vrt_fwd_v7_large_c0-0.csv", fields,
              [dict(tag="v7", budget="large", epoch=0, seed=42, dc=.1, p=p)])
    first = ResultRoutes(tmp_path / "first")
    second = ResultRoutes(tmp_path / "second")
    assert first.cell("v7", "eval", "vrt", 10000)[0][42][0] == .01
    assert second.cell("v7", "eval", "vrt", 10000)[0][42][0] == .9


def test_validation_includes_nested_null_families(tmp_path):
    path = tmp_path / "null/vrt_fwd/vrt_fwd_imgnet_rand_eval_large.csv"
    fields = ["tag", "budget", "epoch", "seed", "dc", "p"]
    record = dict(tag="imgnet_rand_eval", budget="large", epoch=0, seed=42, dc=.1, p=.5)
    write(path, fields, [record])
    assert validate_tree(tmp_path)["records"] == 1
    record["p"] = "nan"
    write(path, fields, [record])
    with pytest.raises(SchemaError, match="is not p"):
        validate_tree(tmp_path)


def test_manifest_verification_detects_changed_payload(tmp_path):
    data = tmp_path / "payload.txt"
    data.write_text("original")
    write(tmp_path / "MANIFEST.csv", ["path", "rows", "bytes", "sha256"],
          [dict(path=data.name, rows="", bytes=data.stat().st_size, sha256=sha256(data))])
    assert manifest_check(tmp_path) == (1, 0)
    data.write_text("tampered")
    with pytest.raises(SchemaError, match="does not match"):
        manifest_check(tmp_path)
