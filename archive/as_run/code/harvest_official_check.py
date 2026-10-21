"""harvest_official_check.py — results of cifar_official_check.py.
1. Rejection rates (p < 0.05) of MMD, ED and VRT on the official and on a random split, for the pixel features (no
   network) and for the random ResNet-18 features extracted in shuffled order.
2. For r18_shuffled: agreement with the null run (same networks, same analysis seeds, official order of extraction),
   repetition by repetition: the share of repetitions with the same decision and the median absolute p difference.
   Agreement means the extraction order plays no role.
Usage:  python harvest_official_check.py [check_dir] [null_vrt_dir] [null_base_dir]
"""
import sys, os, glob, csv, collections
import numpy as np
from scipy.stats import binom

S50 = "/home3/p318770/venvs/overfitting/benchmark_final/s50"
CHK = sys.argv[1] if len(sys.argv) > 1 else f"{S50}/official_check"
VRT_DIR = sys.argv[2] if len(sys.argv) > 2 else f"{S50}/null_cifar_vrt"
BASE_DIR = sys.argv[3] if len(sys.argv) > 3 else f"{S50}/null_cifar_base"

P = collections.defaultdict(dict)          # (features, dataset, split, method) -> {rep: p}
for f in glob.glob(os.path.join(CHK, "official_check_*.csv")):
    if "DRYRUN" in f or os.path.getsize(f) == 0:
        continue
    for r in csv.DictReader(open(f)):
        P[(r["features"], r["dataset"], r["split"], r["method"])].setdefault(int(r["rep"]), float(r["p"]))
if not P:
    sys.exit(f"no result files in {CHK}")

print("rejections at p < 0.05 (calibrated range = central 95% of Binomial(n, 0.05))\n")
print(f"{'features':14s} {'dataset':9s} {'method':6s} {'official split':>20s} {'random split':>20s}")
for feat, ds, m in sorted({(k[0], k[1], k[3]) for k in P}, key=lambda t: (t[0], t[1], ["mmd", "ed", "vrt"].index(t[2]))):
    cells = []
    for s in ("official", "resplit"):
        d = P.get((feat, ds, s, m), {})
        if not d:
            cells.append(f"{'.':>20s}"); continue
        n, k = len(d), sum(p < 0.05 for p in d.values())
        lo, hi = binom.ppf(0.025, n, 0.05), binom.ppf(0.975, n, 0.05)
        cells.append(f"{k}/{n} ({100 * k / n:.1f}%){'*' if k > hi else ('-' if k < lo else ' ')}".rjust(20))
    print(f"{feat:14s} {ds:9s} {m.upper():6s} " + " ".join(cells))

# repetition-by-repetition agreement of r18_shuffled with the null run (official order of extraction)
NULL = collections.defaultdict(dict)       # (dataset, split, method) -> {rep: p}
arch2ds = {"r18_c10": "cifar10", "r18_c100": "cifar100"}
for f in glob.glob(os.path.join(BASE_DIR, "base_null_r18_*_eval_gpu_*.csv")):
    if "DRYRUN" in f:
        continue
    for r in csv.DictReader(open(f)):
        if r["arch"] in arch2ds and r["method"] in ("mmd", "ed"):
            NULL[(arch2ds[r["arch"]], r["split"], r["method"])].setdefault(int(r["rep"]), float(r["p"]))
for f in glob.glob(os.path.join(VRT_DIR, "vrt_null_r18_*_eval_r*.csv")):
    if "DRYRUN" in f:
        continue
    for r in csv.DictReader(open(f)):
        if r["arch"] in arch2ds and r["budget"] == "large":
            NULL[(arch2ds[r["arch"]], r["split"], "vrt")].setdefault(int(r["rep"]), float(r["p"]))
rows = [(ds, s, m) for (feat, ds, s, m) in P if feat == "r18_shuffled"]
if rows:
    print("\nr18_shuffled against the null run (same network and seeds, official extraction order), per repetition:")
    print(f"{'dataset':9s} {'split':9s} {'method':6s} {'reps':>5s} {'same decision':>14s} {'median |dp|':>12s} {'max |dp|':>9s}")
    for ds, s, m in sorted(set(rows)):
        a, b = P[("r18_shuffled", ds, s, m)], NULL.get((ds, s, m), {})
        common = sorted(set(a) & set(b))
        if not common:
            print(f"{ds:9s} {s:9s} {m.upper():6s}  no matching repetitions in the null run"); continue
        dp = np.array([abs(a[r] - b[r]) for r in common])
        same = sum((a[r] < 0.05) == (b[r] < 0.05) for r in common)
        print(f"{ds:9s} {s:9s} {m.upper():6s} {len(common):5d} {f'{same}/{len(common)}':>14s} {np.median(dp):12.4f} {dp.max():9.4f}")
print("\npixels = no network; r18_shuffled = random ResNet-18, images through the network in random order")
