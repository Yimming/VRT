"""harvest_cifar_null.py — single-test rejections of every method on the random-initialization CIFAR nulls, at the 10,000
tier of Table 1 (VRT M = 2,000 / k = 60; MMD, ED, Henze, Schilling and C2ST at 10,000 per split; MMDAgg at 5,000).
Reads the VRT run (cifar_null_vrt.py) and the baseline runs (cifar_null_base.py) and prints two tables, one row per
method and one column per architecture, each cell  rejections/repetitions (percent):
  random split    a new random 50,000 / 10,000 split of the 60,000 images in every repetition: independent null tests,
                  the Type-I estimate
  official split  the official test images in every repetition (only the network and the training draw change): the
                  repetitions share one test sample, so they are not independent; for comparison only
A rejection is p < 0.05 for one test (MMDAgg: its reject decision at 0.05). C2ST is shown twice, p < 0.05 alone and with
the paper's extra rule held-out accuracy > 0.55. '*' marks a count above the central 95 percent range of
Binomial(n, 0.05), the range of a test whose rate is exactly 5 percent (16 to 35 for n = 500), '-' a count below it
(a conservative test, expected for the C2ST rule).
Usage:  python harvest_cifar_null.py [vrt_dir] [base_dir]
"""
import sys, os, glob, csv, collections
import numpy as np
from scipy.stats import binom

S50 = "/home3/p318770/venvs/overfitting/benchmark_final/s50"
VRT_DIR = sys.argv[1] if len(sys.argv) > 1 else f"{S50}/null_cifar_vrt"
BASE_DIR = sys.argv[2] if len(sys.argv) > 2 else f"{S50}/null_cifar_base"
ARCH_ORDER = ["r18_c10", "r18_c100", "r50_c100", "r50_c10", "simclr_c10", "simclr_c100", "frn_c10"]
SHORT = {"r18_c10": "R18 C10", "r18_c100": "R18 C100", "r50_c100": "R50 C100", "r50_c10": "R50 C10",
         "simclr_c10": "SimCLR C10", "simclr_c100": "SimCLR C100", "frn_c10": "R20-FRN C10"}
METHODS = ["VRT", "MMD", "MMDAgg", "C2ST p<.05", "C2ST rule", "ED", "Henze", "Schilling"]

R = collections.defaultdict(dict)        # (split, method, column) -> {rep: rejected}


def add(split, method, col, rep, hit):
    R[(split, method, col)].setdefault(rep, bool(hit))      # first row of a repetition wins


def column(arch, mode):
    return arch if mode == "eval" else f"{arch}/{mode}"


for f in glob.glob(os.path.join(VRT_DIR, "vrt_null_*.csv")):
    if "DRYRUN" in f or os.path.getsize(f) == 0:
        continue
    for r in csv.DictReader(open(f)):
        if r["budget"] != "large":
            continue
        p = float(r["p"])
        if np.isfinite(p):
            add(r["split"], "VRT", column(r["arch"], r["bn_mode"]), int(r["rep"]), p < 0.05)
NAME = {"mmd": "MMD", "ed": "ED", "henze": "Henze", "schilling": "Schilling", "mmdagg": "MMDAgg"}
for f in glob.glob(os.path.join(BASE_DIR, "base_null_*.csv")):
    if "DRYRUN" in f or os.path.getsize(f) == 0:
        continue
    for r in csv.DictReader(open(f)):
        p, col, rep, m = float(r["p"]), column(r["arch"], r["bn_mode"]), int(r["rep"]), r["method"]
        if not np.isfinite(p):
            continue
        if m == "c2st":
            add(r["split"], "C2ST p<.05", col, rep, p < 0.05)
            add(r["split"], "C2ST rule", col, rep, p < 0.05 and float(r["acc"]) > 0.55)
        elif m == "mmdagg":
            add(r["split"], "MMDAgg", col, rep, p >= 0.5)
        else:
            add(r["split"], NAME[m], col, rep, p < 0.05)
if not R:
    sys.exit(f"no result files in {VRT_DIR} or {BASE_DIR}")

cols = sorted({k[2] for k in R}, key=lambda c: (ARCH_ORDER.index(c.split("/")[0]) if c.split("/")[0] in ARCH_ORDER else 99, c))
W = 17
for split, title in (("resplit", "RANDOM SPLIT (independent repetitions: the Type-I estimate)"),
                     ("official", "OFFICIAL SPLIT (official test images in every repetition; comparison only)")):
    if not any(k[0] == split for k in R):
        continue
    print(f"\n{title}, 10,000 per split (MMDAgg 5,000)")
    print(f"{'method':12s}" + "".join(f"{SHORT.get(c.split('/')[0], c) + ('/' + c.split('/')[1] if '/' in c else ''):>{W}s}" for c in cols))
    for m in METHODS:
        cells, any_cell = [], False
        for c in cols:
            d = R.get((split, m, c))
            if not d:
                cells.append(f"{'.':>{W}s}"); continue
            any_cell = True
            n, k = len(d), int(sum(d.values()))
            lo, hi = binom.ppf(0.025, n, 0.05), binom.ppf(0.975, n, 0.05)
            flag = "*" if k > hi else ("-" if k < lo else " ")
            cells.append(f"{f'{k}/{n} ({100 * k / n:.1f}%)' + flag:>{W}s}")
        if any_cell:
            print(f"{m:12s}" + "".join(cells))
print("\nrejection = single test with p < 0.05 (MMDAgg: reject decision at 0.05; C2ST rule: also held-out accuracy > 0.55)")
print("* / - = above / below the central 95% range of Binomial(n, 0.05) (16 to 35 for n = 500); '-' = conservative")
print("R18 = ResNet-18, R50 = ResNet-50, SimCLR = SimCLR ResNet-50 encoder, R20-FRN = ResNet20-FRN; C10/C100 = CIFAR-10/100")
print("architectures of one dataset share the split and the draw of a repetition, so read each column on its own")
