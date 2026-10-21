"""The grouped release and original experiment folders must aggregate identically."""
import csv
from argparse import Namespace
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import exact_perm_p
import results
from results import (Exporter, FAMILY_PATHS, SUMMARY_PATHS, ResultRoutes, SchemaError, compute_onsets,
                     exact_p_cell, family_path, generated_path, grouped_relative_path, manifest_check,
                     result_files, csv_records, schema_for, summaries_report, validate_tree)


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


def group_tree(source, target, deep=False):
    for path in source.rglob("*"):
        if path.is_file():
            rel = path.relative_to(source)
            dst = target / (grouped_relative_path(rel) if deep else
                            Path(FAMILY_PATHS[rel.parts[0]]).joinpath(*rel.parts[1:]))
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)


@pytest.mark.parametrize("deep", [False, True])
def test_grouped_counts_routes_onsets_and_exact_p_equal_legacy(tmp_path, deep):
    original, grouped = tmp_path / "original", tmp_path / "grouped"
    make_results(original)
    group_tree(original, grouped, deep=deep)
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
    section = Path(FAMILY_PATHS["vrt_fwd"]).parts[0]
    (grouped / section / "README.md").write_text("Main results\n")
    (grouped / "MANIFEST.csv").write_text("previous,manifest\n")
    write(grouped / "summaries/onsets.csv", "stat,onset", [("vrt", 1)])
    for relative in SUMMARY_PATHS.values():
        write(grouped / relative, "method,onset", [("vrt", 1)])
    worker = Exporter(str(grouped), str(exported), False, False)
    worker.run()
    assert manifest_check(exported) == (9, 307)
    assert not (exported / "summaries").exists()
    assert not (exported / "README.md").exists()
    assert not (exported / section / "README.md").exists()
    assert not any((exported / relative).exists() for relative in SUMMARY_PATHS.values())
    assert not (exported / "vrt_fwd").exists()
    path = exported / FAMILY_PATHS["null_trtr"] / "knn_v2/knn_imgnet_rand_eval_trtr_n10000_part.csv"
    with path.open(newline="") as f:
        assert next(csv.DictReader(f))["p_henze_exact"] == f"{100 / 2001:.9f}"
    assert validate_tree(exported)["records"] == 307


def test_raw_export_writes_grouped_layout_and_preserves_clean_bytes(tmp_path):
    original, exported, second = tmp_path / "original", tmp_path / "exported", tmp_path / "second"
    make_results(original)
    vrt = original / "vrt_fwd/vrt_fwd_v7_large_c0-2.csv"
    vrt.write_bytes(vrt.read_bytes().replace(b"\r\n", b"\n"))
    Exporter(str(original), str(exported), False, False).run()
    assert manifest_check(exported) == (9, 307)
    assert not any((exported / family).exists() for family in FAMILY_PATHS)
    for path in original.rglob("*.csv"):
        relative = path.relative_to(original)
        destination = exported / grouped_relative_path(relative)
        assert destination.is_file()
        if relative.parts[0] not in {"reversed", "null_trtr"}:
            assert destination.read_bytes() == path.read_bytes()
    assert compute_onsets(original) == compute_onsets(exported)
    Exporter(str(exported), str(second), False, False).run()
    assert manifest_check(second) == (9, 307)
    with (exported / "MANIFEST.csv").open(newline="") as f:
        for entry in csv.DictReader(f):
            assert (exported / entry["path"]).read_bytes() == (second / entry["path"]).read_bytes()


def test_merged_alias_filename_collision_is_never_overwritten(tmp_path):
    original = tmp_path / "original"
    name = "c2st_r50_bntrain_b1500_c0-0.csv"
    for family in ["c2st", "bntrain_c2st"]:
        write(original / family / name, "model,epoch,budget,stat,seed,acc,p",
              [("r50_bntrain", 0, "b1500", "c2st", 42, .6, .01)])
    with pytest.raises(SchemaError, match="output path collision"):
        Exporter(str(original), str(tmp_path / "exported"), False, False).run()


@pytest.mark.parametrize("family", ["vrt_fwd", "c2st", "bntrain_c2st", "mmdagg", "bntrain_mmdagg"])
def test_ambiguous_original_and_grouped_folders_rejected(tmp_path, family):
    (tmp_path / family).mkdir()
    (tmp_path / FAMILY_PATHS[family]).mkdir(parents=True)
    with pytest.raises(SchemaError, match="ambiguous result layout"):
        ResultRoutes(tmp_path)
    with pytest.raises(SchemaError, match="ambiguous result layout"):
        validate_tree(tmp_path)


@pytest.mark.parametrize("family,tag", [("ssl", "cifar10_bntrain"), ("ssl_bntrain", "cifar10")])
@pytest.mark.parametrize("deep", [False, True])
def test_grouped_simclr_wrong_extraction_folder_rejected(tmp_path, family, tag, deep):
    directory = tmp_path / FAMILY_PATHS[family]
    if deep:
        directory = directory / "simclr_resnet50_cifar10/vrt_mmd_ed_c2st/multiple_budgets"
    write(directory / f"ssl_{tag}_fast_c1-100.csv",
          "model,epoch,budget,stat,seed,acc,p", [(tag, 5, "b10000", "vrt", 42, "nan", .5)])
    with pytest.raises(SchemaError, match="belongs in"):
        validate_tree(tmp_path)


def test_grouped_schema_lookup_and_unknown_branch(tmp_path):
    known, _ = schema_for(FAMILY_PATHS["vrt_fwd"], "vrt_fwd_v7_large_c0-0.csv")
    nested, _ = schema_for(FAMILY_PATHS["null_trtr"] + "/knn_v2",
                          "knn_imgnet_rand_eval_trtr_n10000_part.csv")
    assert known["group"] == "vrt_fwd"
    assert nested["group"] == "knn"
    write(tmp_path / "02_regularization_interventions/wrong.csv", "p", [[.5]])
    with pytest.raises(SchemaError, match="unrecognised"):
        validate_tree(tmp_path)


@pytest.mark.parametrize("custom_output", [False, True])
def test_summaries_use_paper_sections_and_include_four_interventions(tmp_path, monkeypatch, custom_output):
    data = tmp_path / "input"
    output = tmp_path / "tables" if custom_output else data
    rows = [dict(tag=tag, mode="eval", stat="vrt", budget=10000, onset=epoch)
            for tag, epoch in [("ablA", 33), ("ablB", ""), ("ablC", 35), ("ablD", "")]]
    monkeypatch.setattr(results, "compute_onsets", lambda _: rows)
    monkeypatch.setattr(results, "_summary_capture", lambda *args: [dict(result="example")])
    monkeypatch.setattr(results, "_summary_synthetic", lambda _: [dict(rejections=1)])
    counts = summaries_report(Namespace(data=data, out=output if custom_output else None))
    assert set(counts) == set(SUMMARY_PATHS)
    assert counts["regularization_interventions.csv"] == 4
    with (output / SUMMARY_PATHS["regularization_interventions.csv"]).open(newline="") as f:
        interventions = list(csv.DictReader(f))
    assert [row["tags"] for row in interventions] == ["ablA", "ablB", "ablC", "ablD"]
    assert [row["VRT"] for row in interventions] == ["33", "null", "35", "null"]
    assert {str(p.relative_to(output)) for p in output.rglob("*.csv")} == set(SUMMARY_PATHS.values())
    assert all(generated_path(relative) for relative in SUMMARY_PATHS.values())
    assert not generated_path("01_main_results/unexpected.csv")


@pytest.mark.parametrize("family,filename,folders", [
    ("vrt_fwd", "vrt_fwd_v7_large_c0-9.csv",
     "resnet20_frn_cifar10/no_batchnorm/points_per_split_10000"),
    ("mmd_p2000", "mmd_r50_bntrain_n1500_c10-19.csv",
     "resnet50_cifar100/train_mode/points_per_split_1500"),
    ("knn_v2", "knn_zoo_c100_m3_n10000_part0.csv",
     "resnet18_cifar100_zoo_model3/eval_mode/points_per_split_10000"),
    ("c2st", "c2st_r50_b1500-10000_c0-9.csv",
     "resnet50_cifar100/eval_mode/multiple_budgets"),
    ("ssl_bntrain", "ssl_cifar100_bntrain_fast_c1-100.csv",
     "simclr_resnet50_cifar100/vrt_mmd_ed_c2st/multiple_budgets"),
    ("ssl_baselines", "sslbase_cifar10_0-99.csv",
     "legacy_runs/simclr_resnet50_cifar10"),
    ("ssl_nll", "ssl_nll_cifar100_c1-100.csv", "simclr_resnet50_cifar100"),
    ("chat_stats", "chatstats_ablD_c0-49.csv",
     "resnet18_cifar10_weight_decay_and_augmentation/eval_mode"),
])
def test_navigation_paths_accept_all_layouts_and_are_idempotent(family, filename, folders):
    historical = Path(family) / filename
    section_flat = Path(FAMILY_PATHS[family]) / filename
    expected = Path(FAMILY_PATHS[family]) / folders / filename
    assert grouped_relative_path(historical) == expected
    assert grouped_relative_path(section_flat) == expected
    assert grouped_relative_path(expected) == expected


@pytest.mark.parametrize("historical", [
    "null/typeI_verbatim_paper_full.csv",
    "null/vrt_fwd/vrt_fwd_imgnet_rand_eval_large.csv",
    "null_trtr/knn_v2/knn_imgnet_rand_eval_trtr_n10000_part.csv",
    "synth/fig4a_clusters.csv",
    "km_logs/km_D64.log",
    "timing/tests_timing_r50_ep61_vrt64.csv",
    "knn_v2/_superseded/knn_r50_n10000_part0.csv",
])
def test_navigation_keeps_nested_controls_and_special_families(historical):
    path = Path(historical)
    expected = Path(FAMILY_PATHS[path.parts[0]]).joinpath(*path.parts[1:])
    assert grouped_relative_path(path) == expected
    assert grouped_relative_path(expected) == expected


def test_score_statistics_npz_moves_with_its_csv_and_exports_verbatim(tmp_path):
    original = tmp_path / "original"
    csv_name = "chatstats_r50_c53-68.csv"
    npz_name = "chatstats_r50_c53-68_chat_s42.npz"
    expected = Path(FAMILY_PATHS["chat_stats"]) / "resnet50_cifar100/eval_mode"
    assert grouped_relative_path(Path("chat_stats") / csv_name) == expected / csv_name
    assert grouped_relative_path(Path("chat_stats") / npz_name) == expected / npz_name
    assert grouped_relative_path(Path(FAMILY_PATHS["chat_stats"]) / npz_name) == expected / npz_name
    assert grouped_relative_path(expected / npz_name) == expected / npz_name
    source = original / "chat_stats" / npz_name
    source.parent.mkdir(parents=True)
    payload = b"NPZ payload copied without interpreting its numerical arrays"
    source.write_bytes(payload)
    exported = tmp_path / "exported"
    Exporter(str(original), str(exported), False, False).run()
    assert (exported / expected / npz_name).read_bytes() == payload
    assert manifest_check(exported) == (1, 0)


def test_recursive_discovery_is_family_scoped_and_excludes_backups(tmp_path):
    family = tmp_path / FAMILY_PATHS["vrt_fwd"]
    first = family / "vrt_fwd_v7_large_c0-0.csv"
    second = family / "resnet20_frn_cifar10/no_batchnorm/points_per_split_10000/vrt_fwd_v7_large_c1-1.csv"
    header = "tag,budget,epoch,seed,dc,p"
    write(first, header, [("v7", "large", 0, 42, .1, .5)])
    write(second, header, [("v7", "large", 1, 42, .1, .01)])
    write(family / "resnet20_frn_cifar10/dup_backup/vrt_fwd_v7_large_c2-2.csv",
          header, [("v7", "large", 2, 42, .1, .001)])
    write(tmp_path / FAMILY_PATHS["reversed"] / "vrt_fwd_v7_large_c3-3.csv",
          header, [("v7", "large", 3, 42, .1, .001)])
    pattern = family / "vrt_fwd_v7_large_c*.csv"
    assert set(result_files(pattern)) == {first.resolve(), second.resolve()}
    assert len(list(csv_records(pattern))) == 2
    routes = ResultRoutes(tmp_path)
    values = routes.read([str(pattern), str(pattern)])
    assert set(values) == {0, 1}
    assert values[0][42][0] == .5 and values[1][42][0] == .01
