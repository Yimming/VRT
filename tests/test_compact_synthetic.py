"""Synthetic protocols and the serial/parallel command-line paths."""
import csv
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import synthetic

# Small-run outputs recorded from the original generators before consolidation.
SINGLE_EXPECTED = [('null_vmf', 'VRT', 300, 3, 0.5714285714285714),
 ('null_vmf', 'MMD', 300, 3, 0.6428571428571429),
 ('null_vmf', 'VRT', 150, 3, 0.21428571428571427),
 ('null_vmf', 'MMD', 150, 3, 0.21428571428571427),
 ('conc_0.3', 'VRT', 300, 3, 0.35714285714285715),
 ('conc_0.3', 'MMD', 300, 3, 0.5),
 ('conc_0.3', 'VRT', 150, 3, 0.07142857142857142),
 ('conc_0.3', 'MMD', 150, 3, 0.21428571428571427),
 ('mean_2deg', 'VRT', 300, 3, 0.7142857142857143),
 ('mean_2deg', 'MMD', 300, 3, 0.5714285714285714),
 ('mean_2deg', 'VRT', 150, 3, 0.21428571428571427),
 ('mean_2deg', 'MMD', 150, 3, 0.14285714285714285),
 ('mix_0.05', 'VRT', 300, 3, 1.0),
 ('mix_0.05', 'MMD', 300, 3, 0.6428571428571429),
 ('mix_0.05', 'VRT', 150, 3, 0.6428571428571429),
 ('mix_0.05', 'MMD', 150, 3, 0.5714285714285714),
 ('S0_null', 'VRT', 300, 3, 0.5714285714285714),
 ('S0_null', 'MMD', 300, 3, 0.7857142857142857),
 ('S0_null', 'NORM-MED', 300, 3, 0.48502994011976047),
 ('S0_null', 'NORM-DISP', 300, 3, 0.49500998003992014),
 ('S0_null', 'VRT', 150, 3, 0.35714285714285715),
 ('S0_null', 'MMD', 150, 3, 0.07142857142857142),
 ('S0_null', 'NORM-MED', 150, 3, 0.17365269461077845),
 ('S0_null', 'NORM-DISP', 150, 3, 0.2894211576846307),
 ('S1_radial_mean', 'VRT', 300, 3, 0.07142857142857142),
 ('S1_radial_mean', 'MMD', 300, 3, 1.0),
 ('S1_radial_mean', 'NORM-MED', 300, 3, 0.001996007984031936),
 ('S1_radial_mean', 'NORM-DISP', 300, 3, 0.6526946107784432),
 ('S1_radial_mean', 'VRT', 150, 3, 0.14285714285714285),
 ('S1_radial_mean', 'MMD', 150, 3, 0.7142857142857143),
 ('S1_radial_mean', 'NORM-MED', 150, 3, 0.001996007984031936),
 ('S1_radial_mean', 'NORM-DISP', 150, 3, 0.26746506986027946),
 ('S2_radial_tail', 'VRT', 300, 3, 0.5714285714285714),
 ('S2_radial_tail', 'MMD', 300, 3, 0.42857142857142855),
 ('S2_radial_tail', 'NORM-MED', 300, 3, 0.07984031936127745),
 ('S2_radial_tail', 'NORM-DISP', 300, 3, 0.001996007984031936),
 ('S2_radial_tail', 'VRT', 150, 3, 0.07142857142857142),
 ('S2_radial_tail', 'MMD', 150, 3, 0.21428571428571427),
 ('S2_radial_tail', 'NORM-MED', 150, 3, 0.5768463073852296),
 ('S2_radial_tail', 'NORM-DISP', 150, 3, 0.001996007984031936),
 ('S3_angular', 'VRT', 300, 3, 0.5714285714285714),
 ('S3_angular', 'MMD', 300, 3, 0.14285714285714285),
 ('S3_angular', 'NORM-MED', 300, 3, 1.0),
 ('S3_angular', 'NORM-DISP', 300, 3, 0.20758483033932135),
 ('S3_angular', 'VRT', 150, 3, 0.6428571428571429),
 ('S3_angular', 'MMD', 150, 3, 0.07142857142857142),
 ('S3_angular', 'NORM-MED', 150, 3, 0.5469061876247505),
 ('S3_angular', 'NORM-DISP', 150, 3, 0.6127744510978044),
 ('S4_angular_plus_radial', 'VRT', 300, 3, 0.5714285714285714),
 ('S4_angular_plus_radial', 'MMD', 300, 3, 0.14285714285714285),
 ('S4_angular_plus_radial', 'NORM-MED', 300, 3, 0.001996007984031936),
 ('S4_angular_plus_radial', 'NORM-DISP', 300, 3, 0.43313373253493015),
 ('S4_angular_plus_radial', 'VRT', 150, 3, 0.6428571428571429),
 ('S4_angular_plus_radial', 'MMD', 150, 3, 0.07142857142857142),
 ('S4_angular_plus_radial', 'NORM-MED', 150, 3, 0.001996007984031936),
 ('S4_angular_plus_radial', 'NORM-DISP', 150, 3, 0.6766467065868264)]

REFERENCE_EXPECTED = [('conc_0.3', 'A', 3, 0.07893401234499553, 0.4),
 ('conc_0.3', 'B', 3, 0.05423106697308966, 0.55),
 ('mean_2deg', 'A', 3, 0.08982147914894067, 0.3),
 ('mean_2deg', 'B', 3, -0.02665437578182983, 0.65),
 ('mixture_0.05', 'A', 3, 0.09722734796630039, 0.15),
 ('mixture_0.05', 'B', 3, -0.027190280492664476, 0.7),
 ('unif_vs_vmf', 'A', 3, -0.01877439018302196, 0.9),
 ('unif_vs_vmf', 'B', 3, -0.30249203134364355, 0.05)]

@pytest.fixture(autouse=True)
def restore_defaults():
    yield
    synthetic.configure()


def test_configure_restores_filtered_experiments():
    synthetic.configure_worker(300, 32, {"S0_null"}, "single-cluster")
    assert synthetic.RADIAL == ["S0_null"]
    assert synthetic.ANGULAR == []
    synthetic.configure()
    assert synthetic.D == 500 and synthetic.N == 5000
    assert len(synthetic.RADIAL) == 5 and len(synthetic.ANGULAR) == 4
    assert np.linalg.norm(synthetic.MU2) == pytest.approx(1.0)
    assert synthetic.MU2[:2] == pytest.approx([np.cos(np.radians(2)), np.sin(np.radians(2))])
    assert synthetic.vrt_setting(5000) == (2000, 60)


def test_single_cluster_rows_match_original_generator():
    synthetic.configure(300, 32)
    with threadpool_limits(limits=1):
        rows, _ = synthetic.single_cluster_rep((3, [300, 150], 13))
    assert rows == SINGLE_EXPECTED
    synthetic.configure_worker(300, 32, {"mean_2deg", "S0_null"}, "single-cluster")
    with threadpool_limits(limits=1):
        selected, _ = synthetic.single_cluster_rep((3, [300, 150], 13))
    assert selected == [row for row in SINGLE_EXPECTED if row[0] in {"mean_2deg", "S0_null"}]


def test_bandwidth_and_distance_results_match_original_generator():
    synthetic.configure(300, 32)
    with threadpool_limits(limits=1):
        a, b = synthetic.make(10, "tight", .05, 200)
        values, sigma = synthetic.mmd_scales(a, b, [.25, .5, 1., 2.], 13, 1000200)
        distances = synthetic.class_distances(a, 200, 10)
    assert values == {.25: 10/14, .5: 9/14, 1.: 10/14, 2.: 10/14}
    assert sigma == pytest.approx(1.3728922329175048, abs=1e-12)
    assert distances == pytest.approx((.15181714026518373, 1.384896399342047), abs=1e-12)


def test_reference_rows_match_original_generator():
    synthetic.configure(300, 32)
    with threadpool_limits(limits=1):
        rows, _ = synthetic.reference_rep((3, 19, list(synthetic.REFERENCE_CONDITIONS)))
    assert len(rows) == len(REFERENCE_EXPECTED)
    for actual, expected in zip(rows, REFERENCE_EXPECTED):
        assert actual[:3] == expected[:3]
        assert actual[3] == pytest.approx(expected[3], abs=1e-12)
        assert actual[4] == expected[4]


def run_cli(tmp_path, experiment, extra=(), name="out.csv"):
    output = tmp_path / name
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = subprocess.run([sys.executable, str(ROOT / "synthetic.py"), experiment,
                             "--smoke", "--out", str(output), *extra],
                            text=True, capture_output=True, env=env, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SMOKE, not paper results" in result.stdout
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(0 < float(row["p"]) <= 1 for row in rows)
    return rows


@pytest.mark.parametrize("experiment,count", [("clusters", 24), ("single-cluster", 28),
                                               ("bandwidth", 10), ("reference", 8)])
def test_cli_smoke(tmp_path, experiment, count):
    assert len(run_cli(tmp_path, experiment)) == count


@pytest.mark.parametrize("experiment", ["single-cluster", "reference"])
def test_spawned_workers_match_serial_with_filter(tmp_path, experiment):
    options = ("--conditions", "mean_2deg")
    serial = run_cli(tmp_path, experiment, options, "serial.csv")
    parallel = run_cli(tmp_path, experiment, (*options, "--jobs", "2"), "parallel.csv")
    assert parallel == serial
    assert len(parallel) == 2


def test_unknown_condition_is_rejected(tmp_path):
    with pytest.raises(SystemExit) as error:
        synthetic.main(["clusters", "--smoke", "--conditions", "null,typo", "--out", str(tmp_path / "bad.csv")])
    assert error.value.code == 2
    assert not (tmp_path / "bad.csv").exists()
