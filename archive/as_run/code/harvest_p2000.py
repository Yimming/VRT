"""LEGACY (camera-ready code release note): onset table of the 10-seed unified-protocol reruns (earlier stage); the published onsets come from tools/make_onsets.py. Kept as run for provenance; not an entry point of the release."""
"""harvest_p2000.py — onset table for the unified-protocol reruns (2,000 permutations, 10 seeds).
Usage: python harvest_p2000.py <ed|mmd> <dir> <N>
  e.g. python harvest_p2000.py ed  /home3/p318770/venvs/overfitting/benchmark_final/ed_p2000  1500
       python harvest_p2000.py mmd /home3/p318770/venvs/overfitting/benchmark_final/mmd_p2000 10000
Median p over seeds per epoch; onset = first epoch from which every later evaluated epoch has median p < 0.05.
Columns: epochs evaluated, first epoch with median p < 0.05, sustained onset, the submission's Table 1 value
for this method (1,500 budget only), and the forward VRT onset at the same budget.
"""
import sys, glob, os, csv, collections
import numpy as np

method, root, N = sys.argv[1], sys.argv[2], sys.argv[3]

PAPER = {  # submission Table 1 (1,500 budget). "--" no sustained detection, "n/t" not tested, "BN" BatchNorm-attributed
    "ed":  {"r50": "n/t", "phase_bs8": "n/t", "phase_bs128": "n/t", "phase_bs1024": "n/t"},
    "mmd": {"r50": "121", "phase_bs8": "150", "phase_bs128": "<=100", "phase_bs1024": "125",
            "zoo_c100_m0": "50-55", "zoo_c100_m1": "50-55", "zoo_c100_m2": "50-55", "zoo_c100_m3": "50-55", "zoo_c100_m4": "50-55"},
}[method]
VRT = {  # forward VRT onset at the matched budget (rebuttal Ftg8 W1 tables; paper protocol for the SimCLR rows)
    "10000": {"r50": 61, "v7": 19, "ablA": 31, "ablB": None, "ablC": 35, "ablD": None, "wp": None, "phase_bs8": 150, "phase_bs128": "<=100", "phase_bs1024": "<=100",
              "zoo_c10_m0": 23, "zoo_c10_m1": 25, "zoo_c10_m2": 22, "zoo_c10_m3": 28, "zoo_c10_m4": 23,
              "zoo_c100_m0": 34, "zoo_c100_m1": 28, "zoo_c100_m2": 29, "zoo_c100_m3": 29, "zoo_c100_m4": 28,
              "simclr_c10": 189, "simclr_c100": 238},
    "1500":  {"r50": 81, "v7": None, "ablA": None, "ablB": None, "ablC": None, "ablD": None, "wp": None,
              "zoo_c10_m0": None, "zoo_c10_m1": None, "zoo_c10_m2": None, "zoo_c10_m3": None, "zoo_c10_m4": None,
              "zoo_c100_m0": 43, "zoo_c100_m1": 41, "zoo_c100_m2": 44, "zoo_c100_m3": 44, "zoo_c100_m4": 42},
}.get(N, {})
ORDER = ["r50", "phase_bs8", "phase_bs128", "phase_bs1024", "v7", "ablA", "ablB", "ablC", "ablD", "wp"] + \
        [f"zoo_c10_m{i}" for i in range(5)] + [f"zoo_c100_m{i}" for i in range(5)] + ["simclr_c10", "simclr_c100", "imgnet_simple", "imgnet_adv"]
VRT.setdefault("imgnet_simple", "null*"); VRT.setdefault("imgnet_adv", "null*")   # paper: train-as-reference VRT null, reversed intermittent

rows = collections.defaultdict(dict)
for f in glob.glob(os.path.join(root, f"{method}_*_n{N}.csv")):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            rows[r["tag"]][(int(r["epoch"]), int(r["seed"]))] = float(r["p"])

def onset(eps, med):
    for i, e in enumerate(eps):
        if all(med[x] < 0.05 for x in eps[i:]):
            return e
    return None

def fmt(x): return "--" if x is None else str(x)
def fmt_vrt(tag): return fmt(VRT[tag]) if tag in VRT else "n/a"   # n/a = VRT not run at this budget for this setting

print(f"=== {method.upper()}, n={N} per split, 2,000 permutations, 10 seeds ===")
print(f"{'setting':13s} {'epochs':>6s} {'first<.05':>9s} {'onset':>6s} {'paper(T1)':>9s} {'VRT same budget':>15s}")
for tag in ORDER:
    d = rows.get(tag)
    if not d:
        print(f"{tag:13s} {'--':>6s} {'(no csv)':>9s}"); continue
    by = collections.defaultdict(list)
    for (e, s), p in d.items():
        by[e].append(p)
    eps = sorted(by)
    med = {e: float(np.median(by[e])) for e in eps}
    first = next((e for e in eps if med[e] < 0.05), None)
    nseeds = min(len(v) for v in by.values())
    flag = "" if nseeds >= 10 else f"  [{nseeds} seeds on some epochs]"
    paper = PAPER.get(tag, "--") if N == "1500" else ""
    print(f"{tag:13s} {len(eps):6d} {fmt(first):>9s} {fmt(onset(eps, med)):>6s} {paper:>9s} {fmt_vrt(tag):>15s}{flag}")
