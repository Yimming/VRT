"""LEGACY (camera-ready code release note): harvest of the 10-seed C2ST / MMDAgg reruns (earlier stage); the published onsets come from tools/make_onsets.py on the 50-seed files. Kept as run for provenance; not an entry point of the release."""
"""harvest_base.py — onsets for the 10-seed C2ST / MMDAgg reruns on the 20 standard settings.
Usage: python harvest_base.py <dir>      (c2st_10seed or mmdagg_10seed)
Decision per epoch over seeds: c2st median p < 0.05 and median held-out accuracy > 0.55 (Table 1 convention);
mmdagg at least 6 of 10 seeds reject. Onset = first evaluated epoch from which every later evaluated epoch rejects.
"""
import sys, glob, os, csv, collections
import numpy as np

root = sys.argv[1]
ORDER = ["r50", "phase_bs8", "phase_bs128", "phase_bs1024", "v7", "ablA", "ablB", "ablC", "ablD", "wp"] + \
        [f"zoo_c10_m{i}" for i in range(5)] + [f"zoo_c100_m{i}" for i in range(5)] + ["imgnet_simple", "imgnet_adv"]
VRT10K = {"r50": 61, "v7": 19, "ablA": 31, "ablC": 35, "phase_bs8": 150, "phase_bs128": "<=100", "phase_bs1024": "<=100",
          "zoo_c10_m0": 23, "zoo_c10_m1": 25, "zoo_c10_m2": 22, "zoo_c10_m3": 28, "zoo_c10_m4": 23,
          "zoo_c100_m0": 34, "zoo_c100_m1": 28, "zoo_c100_m2": 29, "zoo_c100_m3": 29, "zoo_c100_m4": 28,
          "imgnet_simple": "null*", "imgnet_adv": "null*"}

rows = collections.defaultdict(lambda: collections.defaultdict(dict))   # (tag, stat, budget) -> epoch -> seed -> (acc, p)
for f in glob.glob(os.path.join(root, "*.csv")):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            rows[(r["model"], r["stat"], r["budget"])][int(r["epoch"])][int(r["seed"])] = (float(r["acc"]), float(r["p"]))

def decide(stat, vals):
    accs = np.array([a for a, _ in vals]); ps = np.array([p for _, p in vals])
    if stat == "mmdagg":
        return np.mean(ps) >= 0.6
    return np.median(ps) < 0.05 and np.median(accs) > 0.55

def onset(eps, rej):
    for i, e in enumerate(eps):
        if all(rej[x] for x in eps[i:]):
            return e
    return None

f = lambda x: "--" if x is None else str(x)
keys = sorted({(s, b) for (_, s, b) in rows})
for stat, budget in keys:
    print(f"=== {stat} {budget}, 10 seeds ===")
    print(f"{'setting':13s} {'epochs':>6s} {'first':>6s} {'onset':>6s} {'VRT 10k':>8s} {'seeds':>6s}")
    for tag in ORDER:
        d = rows.get((tag, stat, budget))
        if not d:
            print(f"{tag:13s} {'--':>6s} (no rows)"); continue
        eps = sorted(d)
        rej = {e: decide(stat, list(d[e].values())) for e in eps}
        first = next((e for e in eps if rej[e]), None)
        nseeds = min(len(d[e]) for e in eps)
        flag = "" if nseeds >= 10 else f"  [{nseeds} seeds on some epochs]"
        print(f"{tag:13s} {len(eps):6d} {f(first):>6s} {f(onset(eps, rej)):>6s} {f(VRT10K.get(tag)) if tag in VRT10K else '--':>8s} {nseeds:6d}{flag}")
    print()
