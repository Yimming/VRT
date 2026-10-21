"""
harvest_rev.py — summarize reversed-VRT CSVs: median p over seeds per epoch,
sustained onset (alpha=0.05), side by side with forward onsets (rebuttal tables)
and the paper's Table 3 reversed values (paper protocol = large budget).

Usage (cluster):
  python harvest_rev.py /home3/p318770/venvs/overfitting/benchmark_final/reversed
"""
import sys, glob, os, csv, collections
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "."

# forward onsets from the rebuttal matched-budget tables (per model)
FWD = {
    "large": {"r50": 61, "v7": 19, "ablA": 31, "ablB": None, "ablC": 35, "ablD": None, "wp": None,
              "zoo_c10_m0": 23, "zoo_c10_m1": 25, "zoo_c10_m2": 22, "zoo_c10_m3": 28, "zoo_c10_m4": 23,
              "zoo_c100_m0": 34, "zoo_c100_m1": 28, "zoo_c100_m2": 29, "zoo_c100_m3": 29, "zoo_c100_m4": 28},
    "small": {"r50": 81, "v7": None, "ablA": None, "ablB": None, "ablC": None, "ablD": None, "wp": None,
              "zoo_c10_m0": None, "zoo_c10_m1": None, "zoo_c10_m2": None, "zoo_c10_m3": None, "zoo_c10_m4": None,
              "zoo_c100_m0": 43, "zoo_c100_m1": 41, "zoo_c100_m2": 44, "zoo_c100_m3": 44, "zoo_c100_m4": 42},
}
# paper Appendix G Table 3, reversed direction (ranges for zoos)
PAPER_REV = {"r50": "61", "v7": "24", "ablA": "31", "ablB": "null", "ablC": "38", "ablD": "null", "wp": "null",
             "zoo_c10": "23-31 (range)", "zoo_c100": "29-35 (range)"}

def fmt(x): return "ND" if x is None else str(x)

def onset(eps, med):
    for i, e in enumerate(eps):
        if all(med[x] < 0.05 for x in eps[i:]):
            return e
    return None

rows = collections.defaultdict(dict)   # (tag,budget) -> {(epoch,seed): p}
for f in sorted(glob.glob(os.path.join(root, "vrt_rev_*.csv"))):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            rows[(r["tag"], r["budget"])][(int(r["epoch"]), int(r["seed"]))] = float(r["p"])  # dedupe

order = ["r50", "v7", "ablA", "ablB", "ablC", "ablD", "wp"] + \
        [f"zoo_c10_m{i}" for i in range(5)] + [f"zoo_c100_m{i}" for i in range(5)] + \
        ["phase_bs8", "phase_bs128", "phase_bs1024", "imgnet_simple", "imgnet_adv", "simclr_c10", "simclr_c100"]
FWD["large"].update({"phase_bs8": 150, "phase_bs128": "<=100", "phase_bs1024": "<=100", "simclr_c10": 189, "simclr_c100": 238,
                     "imgnet_simple": "null*", "imgnet_adv": "null*"})   # paper protocol == large budget
PAPER_REV.update({"phase_bs8": "150", "phase_bs128": "<=100", "phase_bs1024": "<=100"})
for budget in ["large", "small"]:
    print(f"\n=== budget: {budget} ===")
    print(f"{'setting':14s} {'epochs':>7s} {'rev onset':>10s} {'fwd onset':>10s} {'paper Tab3 rev':>16s}")
    for tag in order:
        d = rows.get((tag, budget))
        if not d:
            print(f"{tag:14s} {'--':>7s} {'(no csv)':>10s}"); continue
        byep = collections.defaultdict(list)
        for (e, s), p in d.items(): byep[e].append(p)
        eps = sorted(byep)
        med = {e: float(np.median(byep[e])) for e in eps}
        on = onset(eps, med)
        nseeds = min(len(v) for v in byep.values())
        paper = PAPER_REV.get(tag, PAPER_REV.get(tag.rsplit("_m", 1)[0], "")) if budget == "large" else ""
        flag = "" if nseeds >= 10 else f"  [only {nseeds} seeds on some epochs]"
        fwd_str = fmt(FWD[budget][tag]) if tag in FWD[budget] else "n/a"
        print(f"{tag:14s} {len(eps):7d} {fmt(on):>10s} {fwd_str:>10s} {paper:>16s}{flag}")
