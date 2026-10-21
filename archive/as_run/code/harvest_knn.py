"""harvest_knn.py — merge knn_*_n<N>_part*.csv (all settings; N = 10000 default, or pass e.g. 1500), median p over seeds per epoch,
sustained onset over the evaluated epochs, side by side with the forward VRT onset at the SAME budget. Usage: python harvest_knn.py <dir> [N]
"""
import sys, glob, os, csv, collections
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "."
N = sys.argv[2] if len(sys.argv) > 2 else "10000"   # budget label in the file names

# forward VRT onset at the MATCHED budget (rebuttal Ftg8 tables): large = 10,000/split, small = 1,500/split
VRT_BY_BUDGET = {
    "10000": {"r50": 61, "v7": 19, "ablA": 31, "ablB": None, "ablC": 35, "ablD": None, "wp": None,
              "zoo_c10_m0": 23, "zoo_c10_m1": 25, "zoo_c10_m2": 22, "zoo_c10_m3": 28, "zoo_c10_m4": 23,
              "zoo_c100_m0": 34, "zoo_c100_m1": 28, "zoo_c100_m2": 29, "zoo_c100_m3": 29, "zoo_c100_m4": 28},
    "1500":  {"r50": 81, "v7": None, "ablA": None, "ablB": None, "ablC": None, "ablD": None, "wp": None,
              "zoo_c10_m0": None, "zoo_c10_m1": None, "zoo_c10_m2": None, "zoo_c10_m3": None, "zoo_c10_m4": None,
              "zoo_c100_m0": 43, "zoo_c100_m1": 41, "zoo_c100_m2": 44, "zoo_c100_m3": 44, "zoo_c100_m4": 42},
}
VRT = VRT_BY_BUDGET.get(N, {})
ORDER = ["r50", "v7", "ablA", "ablB", "ablC", "ablD", "wp"] + \
        [f"zoo_c10_m{i}" for i in range(5)] + [f"zoo_c100_m{i}" for i in range(5)] + \
        ["phase_bs8", "phase_bs128", "phase_bs1024", "imgnet_simple", "imgnet_adv"]
VRT_BY_BUDGET["10000"].update({"phase_bs8": 150, "phase_bs128": "<=100", "phase_bs1024": "<=100", "imgnet_simple": "null*", "imgnet_adv": "null*"})

rows = collections.defaultdict(dict)
for f in glob.glob(os.path.join(root, f"knn_*_n{N}_part*.csv")):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            rows[r["tag"]][(int(r["epoch"]), int(r["seed"]))] = (float(r["p_schilling"]), float(r["p_henze"]))

def onset(eps, med):
    for i, e in enumerate(eps):
        if all(med[x] < 0.05 for x in eps[i:]):
            return e
    return None

def fmt(x): return "ND" if x is None else str(x)

print(f"=== Schilling / Henze at n={N} per split ===")
print(f"{'setting':13s} {'epochs':>6s} {'Schilling':>10s} {'Henze':>7s} {'VRT (same budget)':>18s}")
for tag in ORDER:
    d = rows.get(tag)
    if not d:
        print(f"{tag:13s} {'--':>6s} {'(no csv)':>10s}"); continue
    by = collections.defaultdict(lambda: ([], []))
    for (e, s), (ps, ph) in d.items():
        by[e][0].append(ps); by[e][1].append(ph)
    eps = sorted(by)
    ms = {e: float(np.median(by[e][0])) for e in eps}
    mh = {e: float(np.median(by[e][1])) for e in eps}
    nseeds = min(len(v[0]) for v in by.values())
    flag = "" if nseeds >= 10 else f"  [{nseeds} seeds on some epochs]"
    vrt_str = fmt(VRT[tag]) if tag in VRT else "n/a"
    print(f"{tag:13s} {len(eps):6d} {fmt(onset(eps, ms)):>10s} {fmt(onset(eps, mh)):>7s} {vrt_str:>18s}{flag}")

# detail for the two headline models at the VRT onset region
for tag, lo, hi in [("r50", 55, 130), ("v7", 15, 60)]:
    d = rows.get(tag)
    if not d: continue
    by = collections.defaultdict(lambda: ([], []))
    for (e, s), (ps, ph) in d.items():
        by[e][0].append(ps); by[e][1].append(ph)
    print(f"\n--- {tag}, epochs {lo} to {hi}: median p (Schilling, Henze), every 5th epoch ---")
    for e in sorted(by):
        if lo <= e <= hi and e % 5 == 0:
            print(f"  ep{e:4d}  {np.median(by[e][0]):.4f}  {np.median(by[e][1]):.4f}")
