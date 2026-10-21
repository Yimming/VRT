"""LEGACY (camera-ready code release note): harvest of the 10-seed Energy Distance files (earlier stage, 500 permutations); not used for the published numbers. Kept as run for provenance; not an entry point of the release."""
"""harvest_ed.py — merge ed_*_n1500.csv, median p over the 10 seeds per epoch, sustained onset (alpha=0.05,
all later evaluated epochs reject), next to the submission's Table 1 ED entry and VRT onset.
Usage: python harvest_ed.py <dir>
"""
import sys, glob, os, csv, collections
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "."
# submission Table 1: ED column ("--" = no sustained detection, "n/t" = not tested) and VRT column
PAPER_ED = {"r50": "n/t", "phase_bs8": "n/t", "phase_bs128": "n/t", "phase_bs1024": "n/t"}
PAPER_VRT = {"r50": 61, "v7": 19, "ablA": 31, "ablB": None, "ablC": 35, "ablD": None, "wp": None,
             "phase_bs8": 150, "phase_bs128": "<=100", "phase_bs1024": "<=100",
             "zoo_c10_m0": "22-28", "zoo_c10_m1": "22-28", "zoo_c10_m2": "22-28", "zoo_c10_m3": "22-28", "zoo_c10_m4": "22-28",
             "zoo_c100_m0": "22-34", "zoo_c100_m1": "22-34", "zoo_c100_m2": "22-34", "zoo_c100_m3": "22-34", "zoo_c100_m4": "22-34"}
ORDER = ["r50", "phase_bs8", "phase_bs128", "phase_bs1024", "v7", "ablA", "ablB", "ablC", "ablD", "wp"] + \
        [f"zoo_c10_m{i}" for i in range(5)] + [f"zoo_c100_m{i}" for i in range(5)]

rows = collections.defaultdict(dict)
for f in glob.glob(os.path.join(root, "ed_*_n1500.csv")):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            rows[r["tag"]][(int(r["epoch"]), int(r["seed"]))] = float(r["p"])

def onset(eps, med):
    for i, e in enumerate(eps):
        if all(med[x] < 0.05 for x in eps[i:]):
            return e
    return None

def fmt(x): return "--" if x is None else str(x)

print("=== Energy Distance, n=1,500 per split, 500 permutations, 10 seeds ===")
print(f"{'setting':13s} {'epochs':>6s} {'first<.05':>9s} {'onset':>6s} {'paper ED':>9s} {'VRT (paper)':>12s}")
for tag in ORDER + sorted(set(rows) - set(ORDER)):
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
    print(f"{tag:13s} {len(eps):6d} {fmt(first):>9s} {fmt(onset(eps, med)):>6s} {PAPER_ED.get(tag, '--'):>9s} "
          f"{fmt(PAPER_VRT.get(tag)):>12s}{flag}")

# detail for ResNet-50 around the other baselines' onset (MMD/MMDAgg/C2ST at 121)
d = rows.get("r50")
if d:
    by = collections.defaultdict(list)
    for (e, s), p in d.items():
        by[e].append(p)
    print("\n--- r50 median p, every 5th epoch from 55 to 200 ---")
    print("  " + "  ".join(f"ep{e}:{np.median(by[e]):.3f}" for e in sorted(by) if 55 <= e <= 200 and e % 5 == 0))
