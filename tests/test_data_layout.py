"""The grouped release and original experiment folders must aggregate identically."""
import csv
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import exact_perm_p
from results import (Exporter, FAMILY_PATHS, ResultRoutes, SchemaError, compute_onsets,
                     exact_p_cell, family_path, manifest_check, schema_for, validate_tree)


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header.split(","))
        writer.writerows(rows)


def make_results(root):
    seeds = range(42, 92)
    write(root / "vrt_fwd/vrt_fwd_v7_large_c0-2.csv", "tag,budget,epoch,seed,dc,p",
          [("v7", "large", epoch, seed, .1, .5 if epoch == 0 else .01)
           for epoch in range(3) for seed in seeds])
    write(root / "reversed/vrt_rev_v7_large_c0-2.csv", "tag,budget,direction,epoch,seed,M,k,delta_c,p",
          [("v7", "large", "test_ref", epoch, seed, 2000, 60, -.01,
            .5002 if epoch == 0 else .0470 if seed < 67 else .0530)
           for epoch in range(3) for seed in seeds])
    for family, tag in [("c2st", "r50"), ("bntrain_c2st", "r50_bntrain")]:
        write(root / family / f"c2st_{tag}_b1500_c0-0.csv", "model,epoch,budget,stat,seed,acc,p",
              [(tag, 0, "b1500", "c2st", 42, .6, .01)])
    for family, tag in [("mmdagg", "r50"), ("bntrain_mmdagg", "r50_bntrain")]:
        write(root / family / f"mmdagg_{tag}_b5000_c0-0.csv", "model,epoch,budget,stat,seed,acc,p",
              [(tag, 0, "b5000", "mmdagg", 42, "nan", 1)])
    write(root / "null_trtr/knn_v2/knn_imgnet_rand_eval_trtr_n10000_part.csv",
          "tag,n_samples,k,epoch,seed,p_schilling,p_henze",
          [("imgnet_rand_eval_trtr", 10000, 3, 0, 42, .0412, .0500)])
    for family, tag in [("ssl", "cifar10"), ("ssl_bntrain", "cifar10_bntrain")]:
        write(root / family / f"ssl_{tag}_fast_c1-100.csv", "model,epoch,budget,stat,seed,acc,p",
              [(tag, 5, "b10000", "vrt", 42, "nan", .5)])


def group_tree(source, target):
    for path in source.rglob("*"):
        if path.is_file():
            rel = path.relative_to(source)
            dst = target / FAMILY_PATHS[rel.parts[0]] / Path(*rel.parts[1:])
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)


def test_grouped_counts_routes_onsets_and_exact_p_equal_legacy(tmp_path):
    original, grouped = tmp_path / "original", tmp_path / "grouped"
    make_results(original)
    group_tree(original, grouped)
    before, after = validate_tree(original), validate_tree(grouped)
    assert before == after
    assert after["files"] == 9 and after["records"] == 307
    assert after["cross_file_duplicates"] == 0
    assert validate_tree(grouped, families=["c2st", "bntrain_c2st"])["files"] == 2
    old, new = ResultRoutes(original), ResultRoutes(grouped)
    for tag, mode, method, budget in [("v7", "eval", "vrt", 10000),
                                      ("v7", "eval", "vrt_rev", 10000),
                                      ("r50", "train", "c2st", 1500),
                                      ("r50", "train", "mmdagg", 5000)]:
        # NaN accuracy is absent from these comparisons.
        left, right = old.cell(tag, mode, method, budget), new.cell(tag, mode, method, budget)
        assert {e: {s: v[0] for s, v in ss.items()} for e, ss in left.items()} == {
            e: {s: v[0] for s, v in ss.items()} for e, ss in right.items()}
    exact = exact_p_cell(new, exact_perm_p, "v7", "eval", "vrt_rev", 10000)
    assert exact[1][42] == 94 / 2001
    assert compute_onsets(original) == compute_onsets(grouped)
    assert family_path(grouped, "c2st") == family_path(grouped, "bntrain_c2st")


def test_grouped_export_keeps_paths_and_skips_generated_outputs(tmp_path):
    original, grouped, exported = tmp_path / "original", tmp_path / "grouped", tmp_path / "exported"
    make_results(original)
    group_tree(original, grouped)
    (grouped / "README.md").write_text("Data guide\n")
    (grouped / "benchmarks/README.md").write_text("Benchmarks\n")
    (grouped / "MANIFEST.csv").write_text("previous,manifest\n")
    write(grouped / "summaries/onsets.csv", "stat,onset", [("vrt", 1)])
    worker = Exporter(str(grouped), str(exported), False, False)
    worker.run()
    assert manifest_check(exported) == (9, 307)
    assert not (exported / "summaries").exists()
    assert not (exported / "README.md").exists()
    assert not (exported / "benchmarks/README.md").exists()
    assert not (exported / "vrt_fwd").exists()
    path = exported / FAMILY_PATHS["null_trtr"] / "knn_v2/knn_imgnet_rand_eval_trtr_n10000_part.csv"
    with path.open(newline="") as f:
        assert next(csv.DictReader(f))["p_henze_exact"] == f"{100 / 2001:.9f}"
    assert validate_tree(exported)["records"] == 307


@pytest.mark.parametrize("family", ["vrt_fwd", "c2st", "bntrain_c2st", "mmdagg", "bntrain_mmdagg"])
def test_ambiguous_original_and_grouped_folders_rejected(tmp_path, family):
    (tmp_path / family).mkdir()
    (tmp_path / FAMILY_PATHS[family]).mkdir(parents=True)
    with pytest.raises(SchemaError, match="ambiguous result layout"):
        ResultRoutes(tmp_path)
    with pytest.raises(SchemaError, match="ambiguous result layout"):
        validate_tree(tmp_path)


@pytest.mark.parametrize("family,tag", [("ssl", "cifar10_bntrain"), ("ssl_bntrain", "cifar10")])
def test_grouped_simclr_wrong_extraction_folder_rejected(tmp_path, family, tag):
    write(tmp_path / FAMILY_PATHS[family] / f"ssl_{tag}_fast_c1-100.csv",
          "model,epoch,budget,stat,seed,acc,p", [(tag, 5, "b10000", "vrt", 42, "nan", .5)])
    with pytest.raises(SchemaError, match="belongs in"):
        validate_tree(tmp_path)


def test_grouped_schema_lookup_and_unknown_branch(tmp_path):
    known, _ = schema_for("benchmarks/vrt", "vrt_fwd_v7_large_c0-0.csv")
    nested, _ = schema_for("calibration/imagenet/disjoint_training/knn_v2",
                          "knn_imgnet_rand_eval_trtr_n10000_part.csv")
    assert known["group"] == "vrt_fwd"
    assert nested["group"] == "knn"
    write(tmp_path / "benchmarks/unknown/wrong.csv", "p", [[.5]])
    with pytest.raises(SchemaError, match="unrecognised"):
        validate_tree(tmp_path)
