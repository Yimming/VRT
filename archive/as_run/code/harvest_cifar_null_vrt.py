"""harvest_cifar_null_vrt.py — VRT rejections on the random-initialization CIFAR nulls written by cifar_null_vrt.py.

One line per architecture, BatchNorm mode, split and budget:
  n         repetitions found
  rej       repetitions with p < 0.05, count and percent
  95% CI    exact (Clopper-Pearson) interval of the rejection rate
  calibrated range   central 95 percent range of the count for a test with rate exactly 0.05, Binomial(n, 0.05)
                     (16 to 35 for n = 500); OUTSIDE marks a count outside it
  p<.01, p<.10       rejection percentages at two other levels; median p (about 0.5 for a calibrated test)
split resplit  = new random split of the 60,000 images in every repetition: independent null tests, the Type-I estimate
split official = official test images in every repetition (only network and training draw change): repetitions share one
                 test sample, so they are not independent; shown for comparison
Usage:  python harvest_cifar_null_vrt.py [out_dir]
"""
import sys, os, glob, csv, collections
import numpy as np
from scipy.stats import binom, beta

BF = sys.argv[1] if len(sys.argv) > 1 else "/home3/p318770/venvs/overfitting/benchmark_final/s50/null_cifar_vrt"
ARCH_ORDER = ["r18_c10", "r18_c100", "r50_c100", "r50_c10", "simclr_c10", "simclr_c100", "frn_c10"]
NAME = {"r18_c10": "ResNet-18 CIFAR-10", "r18_c100": "ResNet-18 CIFAR-100", "r50_c100": "ResNet-50 CIFAR-100",
        "r50_c10": "ResNet-50 CIFAR-10", "simclr_c10": "SimCLR enc. CIFAR-10", "simclr_c100": "SimCLR enc. CIFAR-100",
        "frn_c10": "ResNet20-FRN CIFAR-10"}

P = collections.defaultdict(dict)           # (arch, mode, split, budget) -> {rep: p}
bad = collections.Counter(); dup = collections.Counter()
files = sorted(f for f in glob.glob(os.path.join(BF, "vrt_null_*.csv")) if "DRYRUN" not in f)
for f in files:
    if os.path.getsize(f) == 0:
        continue
    for r in csv.DictReader(open(f)):
        key = (r["arch"], r["bn_mode"], r["split"], r["budget"])
        p = float(r["p"])
        if not np.isfinite(p):
            bad[key] += 1; continue
        rep = int(r["rep"])
        if rep in P[key]:
            dup[key] += 1; continue          # a repetition written twice: keep the first
        P[key][rep] = p
if not P:
    sys.exit(f"no result files in {BF}")


def cp(k, n, a=0.05):
    return (beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0), (beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0)


def order(key):
    a, m, s, b = key
    return (ARCH_ORDER.index(a) if a in ARCH_ORDER else 99, m, s != "resplit", b != "large")


print(f"{len(files)} files in {BF}\n")
print(f"{'model':22s} {'BN':5s} {'split':8s} {'budget':6s} {'n':>4s} {'rej':>11s} {'95% CI':>13s} "
      f"{'calibrated':>11s} {'p<.01':>6s} {'p<.10':>6s} {'med p':>6s}")
pool = collections.defaultdict(lambda: [0, 0])
for key in sorted(P, key=order):
    a, m, s, b = key
    ps = np.array(list(P[key].values())); n = len(ps); k = int((ps < 0.05).sum())
    lo, hi = cp(k, n)
    rlo, rhi = int(binom.ppf(0.025, n, 0.05)), int(binom.ppf(0.975, n, 0.05))
    note = ("" if rlo <= k <= rhi else "  OUTSIDE") + (f"  {bad[key]} non-finite" if bad[key] else "") + \
           (f"  {dup[key]} duplicate rows" if dup[key] else "")
    budget = "10,000" if b == "large" else "1,500"
    print(f"{NAME.get(a, a):22s} {m:5s} {s:8s} {budget:>6s} {n:4d} {k:4d} ({100 * k / n:4.1f}%) "
          f"[{100 * lo:4.1f},{100 * hi:5.1f}] {rlo:5d} to {rhi:<3d} {100 * np.mean(ps < 0.01):5.1f}% "
          f"{100 * np.mean(ps < 0.10):5.1f}% {np.median(ps):6.3f}{note}")
    pool[(m, s, b)][0] += k; pool[(m, s, b)][1] += n
print("\nall architectures together")
for (m, s, b), (k, n) in sorted(pool.items(), key=lambda t: (t[0][0], t[0][1] != "resplit", t[0][2] != "large")):
    lo, hi = cp(k, n)
    print(f"  {m:5s} {s:8s} {'10,000' if b == 'large' else '1,500':>6s}  {k}/{n} ({100 * k / n:.1f}%)  95% CI "
          f"[{100 * lo:.1f}, {100 * hi:.1f}]  calibrated range {int(binom.ppf(0.025, n, 0.05))} to {int(binom.ppf(0.975, n, 0.05))}")
print("\nresplit = new random 50,000 / 10,000 split of the 60,000 images in every repetition (independent tests, Type-I "
      "estimate);\nofficial = official test images in every repetition (shared test sample, not independent; comparison only)")
