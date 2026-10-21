"""LEGACY (camera-ready code release note): 10-seed train-mode BatchNorm harvest (earlier stage); the published Table 2 comes from tools/make_onsets.py / harvest_bn_table.py. Kept as run for provenance; not an entry point of the release."""
"""harvest_bntrain.py — Section 4.4 check on the 15 BatchNorm CIFAR settings: sustained onset of every method
under eval-mode BatchNorm (running statistics, the Table 1 protocol) -> under train-mode BatchNorm (per-batch
statistics, the rebuttal re-extraction <setting>_bntrain), same images, same seeds, same protocol.
Usage: python harvest_bntrain.py [benchmark_final dir]
Cells read A -> B (eval -> train). "--" no sustained detection, "." not computed. Onset = first epoch from which
the median p over the 10 seeds is below 0.05 at every later evaluated epoch. VRT eval-mode values are the
verified unified-protocol onsets (rebuttal Ftg8 tables); VRT train-mode comes from vrt_fwd / ksweep15 on the
_bntrain features.
"""
import sys, glob, os, csv, collections
import numpy as np

BF = sys.argv[1] if len(sys.argv) > 1 else "/home3/p318770/venvs/overfitting/benchmark_final"
SETTINGS = ["r50", "ablA", "ablB", "ablC", "ablD"] + [f"zoo_c10_m{i}" for i in range(5)] + [f"zoo_c100_m{i}" for i in range(5)]
VRT_EVAL = {
    "10000": {"r50": 61, "ablA": 31, "ablB": None, "ablC": 35, "ablD": None,
              "zoo_c10_m0": 23, "zoo_c10_m1": 25, "zoo_c10_m2": 22, "zoo_c10_m3": 28, "zoo_c10_m4": 23,
              "zoo_c100_m0": 34, "zoo_c100_m1": 28, "zoo_c100_m2": 29, "zoo_c100_m3": 29, "zoo_c100_m4": 28},
    "1500":  {"r50": 81, "ablA": None, "ablB": None, "ablC": None, "ablD": None,
              "zoo_c10_m0": None, "zoo_c10_m1": None, "zoo_c10_m2": None, "zoo_c10_m3": None, "zoo_c10_m4": None,
              "zoo_c100_m0": 43, "zoo_c100_m1": 41, "zoo_c100_m2": 44, "zoo_c100_m3": 44, "zoo_c100_m4": 42},
}


def read_p(pattern, pcol, tagcol=None, tag=None):
    """epoch -> {seed: p} over all files matching the pattern (empty dict if none)."""
    d = collections.defaultdict(dict)
    for f in glob.glob(pattern):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if tagcol and r[tagcol] != tag:
                    continue
                d[int(r["epoch"])][int(r["seed"])] = float(r[pcol])
    return d


def onset(d):
    if not d:
        return "."
    eps = sorted(d)
    med = {e: float(np.median(list(d[e].values()))) for e in eps}
    for i, e in enumerate(eps):
        if all(med[x] < 0.05 for x in eps[i:]):
            return str(e)
    return "--"


def fmt(v):
    return "--" if v is None else str(v)


def cell(a, b):
    return f"{a} -> {b}"


for N in ("10000", "1500"):
    print(f"=== BatchNorm CIFAR settings, n={N} per split: sustained onset, eval-mode BN -> train-mode BN (10 seeds, 2,000 permutations) ===")
    cols = ["VRT", "MMD norm", "ED norm", "Henze", "Schilling", "MMD raw", "ED raw"]
    print(f"{'setting':12s} " + " ".join(f"{c:>12s}" for c in cols))
    for s in SETTINGS:
        t = s + "_bntrain"; T = f"{BF}/bntrain"
        if N == "10000":
            vrt_tr = onset(read_p(f"{T}/vrt_fwd/vrt_fwd_{t}_large.csv", "p"))
        else:
            vrt_tr = onset(read_p(f"{T}/ksweep15/ksweep15_{t}_*.csv", "p", tagcol="model", tag=t))
        row = [cell(fmt(VRT_EVAL[N][s]), vrt_tr),
               cell(onset(read_p(f"{BF}/mmd_p2000/mmd_{s}_n{N}.csv", "p")),     onset(read_p(f"{T}/mmd_p2000/mmd_{t}_n{N}.csv", "p"))),
               cell(onset(read_p(f"{BF}/ed_p2000/ed_{s}_n{N}.csv", "p")),       onset(read_p(f"{T}/ed_p2000/ed_{t}_n{N}.csv", "p"))),
               cell(onset(read_p(f"{BF}/knn_v2/knn_{s}_n{N}_part*.csv", "p_henze")),     onset(read_p(f"{T}/knn_v2/knn_{t}_n{N}_part*.csv", "p_henze"))),
               cell(onset(read_p(f"{BF}/knn_v2/knn_{s}_n{N}_part*.csv", "p_schilling")), onset(read_p(f"{T}/knn_v2/knn_{t}_n{N}_part*.csv", "p_schilling"))),
               cell(onset(read_p(f"{BF}/mmd_raw_p2000/mmd_{s}_n{N}.csv", "p")), onset(read_p(f"{T}/mmd_raw_p2000/mmd_{t}_n{N}.csv", "p"))),
               cell(onset(read_p(f"{BF}/ed_raw_p2000/ed_{s}_n{N}.csv", "p")),   onset(read_p(f"{T}/ed_raw_p2000/ed_{t}_n{N}.csv", "p")))]
        print(f"{s:12s} " + " ".join(f"{c:>12s}" for c in row))
    print()
print("A -> B: onset on eval-mode features -> onset on train-mode features. '--' no sustained detection, '.' not computed yet.")
