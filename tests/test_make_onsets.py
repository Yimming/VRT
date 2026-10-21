"""
Smoke test of results.py onsets on a synthetic result folder: one model (v7) with forward VRT at the large budget
(every epoch 0 to 5 with the 50 analysis seeds), a reversed-VRT file with four-decimal p, and a kNN file. Checks the
sustained-onset rule, the exact-p recovery (0.0470/0.0530 median), the gap and tail counts, and the table printout.
"""
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOL = os.path.join(ROOT, "results.py")
SEEDS = list(range(42, 92))


def write_vrt(path, tag, budget, p_by_epoch, header=("tag", "budget", "epoch", "seed", "dc", "p")):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for ep, p in p_by_epoch.items():
            for s in SEEDS:
                w.writerow([tag, budget, ep, s, "+0.1", f"{p:.6f}"])


def test_make_onsets_synthetic(tmp_path):
    data = tmp_path / "data"
    # forward VRT: epochs 0..5, rejects from epoch 3 on, epoch 1 a false alarm (so the sustained onset is 3, not 1)
    write_vrt(data / "vrt_fwd" / "vrt_fwd_v7_large_c0-5.csv", "v7", "large",
              {0: 0.5, 1: 0.01, 2: 0.3, 3: 0.01, 4: 0.02, 5: 0.001})      # six-decimal p, any value in (0, 1]
    # reversed VRT at epochs 0, 2, 4 (epoch 1, 3 and 5 not evaluated): half the seeds 0.0470, half 0.0530 at epoch 2
    # (four-decimal values as the as-run runner wrote them; make_onsets validates the files first)
    os.makedirs(data / "reversed")
    with open(data / "reversed" / "vrt_rev_v7_large_c0-5.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tag", "budget", "direction", "epoch", "seed", "M", "k", "delta_c", "p"])
        for ep, ps in {0: [0.5002] * 50, 2: [0.0470] * 25 + [0.0530] * 25, 4: [0.0010] * 50}.items():   # on the 1/2001 lattice
            for s, p in zip(SEEDS, ps):
                w.writerow(["v7", "large", "test_ref", ep, s, 2000, 60, "-0.01", f"{p:.4f}"])
    r = subprocess.run([sys.executable, TOOL, "onsets", "--data", str(data), "--out", str(tmp_path / "onsets.csv")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = {(x["stat"], x["budget"]): x for x in csv.DictReader(open(tmp_path / "onsets.csv"))}
    fwd = rows[("vrt", "10000")]
    assert fwd["onset"] == "3" and fwd["n_epochs"] == "6" and fwd["gap_before_onset"] == "0" and fwd["unevaluated_after_onset"] == "0"
    rev = rows[("vrt_rev", "10000")]
    assert rev["onset"] == "2"                               # exact p: median (94 + 106)/2/2001 = 0.049975 rejects
    assert rev["gap_before_onset"] == "1"                    # epoch 1 lies between the last non-rejecting epoch 0 and the onset 2
    assert rev["unevaluated_after_onset"] == "2"             # epochs 3 and 5
    assert "ResNet20-FRN (CIFAR-10)          3 | missing" in r.stdout
    # with the stored four-decimal values the median at epoch 2 is 0.0500 exactly and the onset moves to 4
    r2 = subprocess.run([sys.executable, TOOL, "onsets", "--data", str(data), "--out", str(tmp_path / "o2.csv"), "--stored_p"],
                        capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    rows2 = {(x["stat"], x["budget"]): x for x in csv.DictReader(open(tmp_path / "o2.csv"))}
    assert rows2[("vrt_rev", "10000")]["onset"] == "4" and rows2[("vrt_rev", "10000")]["borderline_epochs"] == "2"
    # an invalid value stops the run instead of counting as "not rejected"
    with open(data / "vrt_fwd" / "vrt_fwd_v7_large_c0-5.csv", "a") as f:
        f.write("v7,large,6,42,+0.1,nan\n")
    r3 = subprocess.run([sys.executable, TOOL, "onsets", "--data", str(data), "--out", str(tmp_path / "o3.csv")], capture_output=True, text=True)
    assert r3.returncode != 0 and "not valid" in r3.stderr
    r4 = subprocess.run([sys.executable, TOOL, "onsets", "--data", str(data), "--out", str(tmp_path / "o4.csv"), "--no_validate"],
                        capture_output=True, text=True)
    assert r4.returncode == 0, r4.stderr                      # epoch 6 has one seed only: a partial epoch, ignored
    # make the row complete with 50 seeds of NaN and check the decision guard
    with open(data / "vrt_fwd" / "vrt_fwd_v7_large_c0-5.csv", "a") as f:
        for s in SEEDS[1:]:
            f.write(f"v7,large,6,{s},+0.1,nan\n")
    r5 = subprocess.run([sys.executable, TOOL, "onsets", "--data", str(data), "--out", str(tmp_path / "o5.csv"), "--no_validate"],
                        capture_output=True, text=True)
    assert r5.returncode != 0 and "outside (0, 1]" in r5.stderr
