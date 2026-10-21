"""
Edge cases of the data tools found in the second review: inconsistent count / p / exact columns, negative metadata,
integer keys ('042' equals '42', 'oops' is an error), rows with another protocol (direction, M, k, n_perm) refused by
make_onsets, cells without a complete epoch listed instead of dropped, the export written through a partial folder
(an empty pre-existing destination stays empty after a refusal), an empty source refused with a message, and the
runners refusing a sample budget larger than the split.
"""
import csv
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from results import SCHEMAS, SchemaError, read_rows, validate_rows, validate_tree, protocol_for  # noqa: E402

EXPORT = "export"
ONSETS = "onsets"
SEEDS = list(range(42, 92))
REV_HEADER = ["tag", "budget", "direction", "epoch", "seed", "M", "k", "delta_c", "p", "count", "n_perm", "p_exact"]


def write(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def check_file(family, path):
    schema = SCHEMAS[family]
    import re
    m = re.match(schema["pattern"], os.path.basename(path))
    header, rows, torn, _, _ = read_rows(path)
    return validate_rows(schema, m.groupdict(), header, rows, path, protocol=protocol_for(schema, m.groupdict()))


def test_count_p_and_exact_columns_must_agree(tmp_path):
    good = ["v7", "large", "test_ref", 0, 42, 2000, 60, "-0.01", "0.500250", 1000, 2000, f"{1001 / 2001:.9f}"]
    path = str(tmp_path / "vrt_rev_v7_large_c0-9.csv")
    write(path, REV_HEADER, [good])
    assert len(check_file("reversed", path)[1]) == 1
    for bad, msg in [(dict(p="0.050000", count=1000), "disagrees"),                 # p does not match count
                     (dict(p="1.000000", count=-2, n_perm=-2), "protocol value|outside"),   # negative metadata
                     (dict(p="1.000000", count=-2), "outside"),                      # negative count, valid n_perm
                     (dict(p_exact="0.010000000"), "does not match"),                # exact column contradicts p
                     (dict(count=2001), "outside")]:                                 # count above n_perm
        row = dict(zip(REV_HEADER, good)); row.update(bad)
        write(path, REV_HEADER, [[row[c] for c in REV_HEADER]])
        with pytest.raises(SchemaError, match=msg):
            check_file("reversed", path)


def test_integer_keys_and_protocol_values(tmp_path):
    path = str(tmp_path / "vrt_rev_v7_large_c0-9.csv")
    base = ["v7", "large", "test_ref", 0, 42, 2000, 60, "-0.01", "0.0470"]
    header = REV_HEADER[:9]
    write(path, header, [base, ["v7", "large", "test_ref", "0", "042", 2000, 60, "-0.02", "0.0530"]])
    _, records = check_file("reversed", path)
    assert records[0][0] == records[1][0]                   # '042' and '42' are one key
    write(path, header, [["v7", "large", "test_ref", 0, "oops", 2000, 60, "-0.01", "0.0470"]])
    with pytest.raises(SchemaError, match="not an integer"):
        check_file("reversed", path)
    for col, val in [("direction", "train_ref"), ("M", 500), ("k", 15)]:
        row = dict(zip(header, base)); row[col] = val
        write(path, header, [[row[c] for c in header]])
        with pytest.raises(SchemaError, match="protocol value"):
            check_file("reversed", path)
    # the same rows pass when the protocol check is off, and the key keeps direction, M and k apart
    import re
    schema = SCHEMAS["reversed"]; m = re.match(schema["pattern"], os.path.basename(path))
    h, rows, *_ = read_rows(path)
    assert len(validate_rows(schema, m.groupdict(), h, rows, path)[1]) == 1
    tree = tmp_path / "tree"
    write(str(tree / "reversed" / "vrt_rev_v7_large_c0-9.csv"), header, [base])
    write(str(tree / "mmd_p2000" / "mmd_v7_n10000_c0-9.csv"), ["tag", "n_samples", "n_perm", "epoch", "seed", "mmd2", "p"],
          [["v7", 10000, 500, 0, 42, "1e-3", "0.5"]])
    with pytest.raises(SchemaError, match="protocol value"):
        validate_tree(str(tree))
    assert validate_tree(str(tree), protocol=False)["records"] == 2
    os.makedirs(tmp_path / "empty_tree_that_exists")
    with pytest.raises(SchemaError, match="no result file"):
        validate_tree(str(tmp_path / "empty_tree_that_exists"))


def run(tool, *args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "results.py"), tool, *args], capture_output=True, text=True)


def test_make_onsets_lists_incomplete_cells_and_refuses_other_protocols(tmp_path):
    data = tmp_path / "data"
    write(str(data / "vrt_fwd" / "vrt_fwd_v7_large_c0-5.csv"), ["tag", "budget", "epoch", "seed", "dc", "p"],
          [["v7", "large", ep, s, "+0.1", "0.5"] for ep in range(3) for s in SEEDS])
    # a reversed cell whose every epoch lacks one seed: listed with a status, not dropped
    write(str(data / "reversed" / "vrt_rev_v7_large_c0-5.csv"), REV_HEADER[:9],
          [["v7", "large", "test_ref", ep, s, 2000, 60, "-0.01", "0.0470"] for ep in range(2) for s in SEEDS[:-1]])
    r = run(ONSETS, "--data", str(data), "--out", str(tmp_path / "o.csv"))
    assert r.returncode == 0, r.stderr
    rows = {(x["stat"], x["budget"]): x for x in csv.DictReader(open(tmp_path / "o.csv"))}
    assert rows[("vrt_rev", "10000")]["status"].startswith("no complete epoch") and rows[("vrt_rev", "10000")]["n_epochs"] == "0"
    assert "incomplete: v7 eval vrt_rev 10000" in r.stdout
    assert rows[("vrt", "10000")]["status"].startswith("null")
    # a reversed file with another M is refused by the protocol check and merged by nothing
    write(str(data / "reversed" / "vrt_rev_v7_large_c6-9.csv"), REV_HEADER[:9],
          [["v7", "large", "test_ref", 6, s, 500, 60, "-0.01", "0.0470"] for s in SEEDS])
    r = run(ONSETS, "--data", str(data), "--out", str(tmp_path / "o2.csv"))
    assert r.returncode != 0 and "protocol value" in r.stderr
    # new-format rows: the exact p comes from the count, not from a lattice assumption
    write(str(data / "reversed" / "vrt_rev_v7_large_c6-9.csv"), REV_HEADER[:11],
          [["v7", "large", "test_ref", 6, s, 2000, 60, "-0.01", "0.049975", 99, 2000] for s in SEEDS])
    r = run(ONSETS, "--data", str(data), "--out", str(tmp_path / "o3.csv"))
    assert r.returncode == 0, r.stderr
    rows = {(x["stat"], x["budget"]): x for x in csv.DictReader(open(tmp_path / "o3.csv"))}
    assert rows[("vrt_rev", "10000")]["onset"] == "6" and rows[("vrt_rev", "10000")]["n_epochs"] == "1"


def test_export_uses_a_partial_folder_and_refuses_an_empty_source(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    write(str(src / "vrt_fwd" / "vrt_fwd_v7_large_c0-9.csv"), ["tag", "budget", "epoch", "seed", "dc", "p"],
          [["v7", "large", 0, 42, "+0.1", "nan"]])
    os.makedirs(dst)                                        # pre-existing and empty: allowed, and must stay empty on refusal
    r = run(EXPORT, "--src", str(src), "--dst", str(dst))
    assert r.returncode != 0 and os.listdir(dst) == [] and not any(p.name.startswith("dst.partial") for p in tmp_path.iterdir())
    write(str(src / "vrt_fwd" / "vrt_fwd_v7_large_c0-9.csv"), ["tag", "budget", "epoch", "seed", "dc", "p"],
          [["v7", "large", 0, 42, "+0.1", "0.5"]])
    r = run(EXPORT, "--src", str(src), "--dst", str(dst))
    assert r.returncode == 0, r.stderr
    assert sorted(os.listdir(dst)) == ["01_main_results", "EXCLUDED.txt", "MANIFEST.csv"]
    empty = tmp_path / "nothing"; os.makedirs(empty)
    r = run(EXPORT, "--src", str(empty), "--dst", str(tmp_path / "dst2"))
    assert r.returncode != 0 and "no files found" in r.stderr and not (tmp_path / "dst2").exists()


def test_runners_refuse_a_budget_larger_than_the_split():
    import numpy as np
    from evaluate import prepare_features

    train = np.ones((2, 3))
    test = np.ones((4, 3))
    for method in ("vrt", "reversed", "mmd", "ed", "schilling", "henze", "c2st", "mmdagg"):
        with pytest.raises(ValueError, match="at least 3 points"):
            prepare_features(train, test, method, 3, "supervised")
