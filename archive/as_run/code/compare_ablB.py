"""LEGACY (camera-ready code release note): Configuration B comparison between the 10-seed and the 50-seed forward VRT (transition stage). Kept as run for provenance; not an entry point of the release."""
"""compare_ablB.py — Configuration B, forward VRT at 10,000 per split: per-epoch median p under eval-mode and
train-mode BatchNorm, with the 10 published seeds (42-51) and with all seeds present, plus the sustained onset
under each. Usage: python compare_ablB.py [benchmark_final dir] [first epoch to print, default 30]
"""
import sys, csv, collections
import numpy as np

BF = sys.argv[1] if len(sys.argv) > 1 else "/home3/p318770/venvs/overfitting/benchmark_final"
FROM = int(sys.argv[2]) if len(sys.argv) > 2 else 30
PUB = set(range(42, 52))
FILES = {"eval": f"{BF}/vrt_fwd/vrt_fwd_ablB_large.csv", "train": f"{BF}/bntrain/vrt_fwd/vrt_fwd_ablB_bntrain_large.csv"}


def onset(med):
    eps = sorted(med)
    for i, e in enumerate(eps):
        if all(med[x] < 0.05 for x in eps[i:]):
            return e
    return None


for mode, f in FILES.items():
    d = collections.defaultdict(dict)
    try:
        for r in csv.DictReader(open(f)):
            d[int(r["epoch"])][int(r["seed"])] = float(r["p"])
    except FileNotFoundError:
        print(f"{mode}: no file yet ({f})"); continue
    med10 = {e: float(np.median([p for s, p in v.items() if s in PUB])) for e, v in d.items() if any(s in PUB for s in v)}
    medall = {e: float(np.median(list(v.values()))) for e, v in d.items()}
    nmax = max(len(v) for v in d.values())
    print(f"=== Config B, {mode}-mode BN: sustained onset {onset(med10)} with 10 seeds, {onset(medall)} with up to {nmax} seeds ===")
    print(f"{'epoch':>5s} {'med p (10)':>10s} {'<.05 /10':>8s} {'med p (all)':>11s} {'<.05 /all':>9s}")
    for e in sorted(d):
        if e < FROM:
            continue
        v = d[e]; pub = [p for s, p in v.items() if s in PUB]
        print(f"{e:5d} {med10.get(e, float('nan')):10.3f} {sum(p < 0.05 for p in pub):5d}/{len(pub):<2d} {medall[e]:11.3f} {sum(p < 0.05 for p in v.values()):5d}/{len(v):<3d}")
    print()
