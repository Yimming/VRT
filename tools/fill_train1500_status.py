"""
fill_train1500_status.py — progress of the train-mode 1,500-point fill (camera-ready #27102, October 16).

The appendix tier at 1,500 points per split was complete in eval mode (24 models, five tests) but not in train mode:
C2ST lacked ResNet-50 and MMDAgg had only the Phase Zoo and ImageNet models. fill_train1500_cpu.txt (49 CPU tasks) adds
  C2ST   r50_bntrain b1500, epochs 0 to 200                      -> bntrain_c2st/c2st_r50_bntrain_b1500_c0-200.csv
  MMDAgg b1500 on the stored train-mode features of r50 (0-200), ablA-D (0-49), zoo_c10_m0-4 (0-50), zoo_c100_m0-4 (0-60)
                                                                  -> bntrain_mmdagg/mmdagg_<tag>_bntrain_b1500_c<lo>-<hi>.csv
  MMDAgg b1500 for the two SimCLR models in train mode on the every-5th grid (ssl_s50_fill.py --group slow --bn_train)
                                                                  -> ssl_bntrain/ssl_<ds>_bntrain_slow_c<lo>-<hi>.csv
This script prints, per cell, the epochs that have all 50 seeds against the expected count, and for the SimCLR grid the
current sustained onset under the MMDAgg rule (at least 60 percent of the seeds rejecting at that epoch and every later
evaluated epoch). After the grid, find_grid_gaps.py --tasks writes the gap and tail tasks for the SimCLR cells.
Usage (cluster, code folder):  python fill_train1500_status.py
"""
import collections, csv, glob, os

B = os.environ.get("S50", "/home3/p318770/venvs/overfitting/benchmark_final/s50")
EXPECT = {"r50_bntrain": range(0, 201), "ablA_bntrain": range(0, 50), "ablB_bntrain": range(0, 50), "ablC_bntrain": range(0, 50),
          "ablD_bntrain": range(0, 50)}
EXPECT.update({f"zoo_c10_m{i}_bntrain": range(0, 51) for i in range(5)})
EXPECT.update({f"zoo_c100_m{i}_bntrain": range(0, 61) for i in range(5)})


def seeds_per_epoch(pattern, model=None, stat=None, budget="b1500"):
    d = collections.defaultdict(set)
    vals = collections.defaultdict(dict)
    for f in glob.glob(pattern):
        for r in csv.DictReader(open(f)):
            if r.get("budget") != budget or (stat and r.get("stat") != stat) or (model and r.get("model") != model):
                continue
            try:
                e, s = int(r["epoch"]), int(r["seed"])
            except (KeyError, ValueError):
                continue
            d[e].add(s); vals[e][s] = float(r["p"])
    return d, vals


def line(name, d, expected):
    full = sorted(e for e in d if len(d[e]) >= 50 and e in expected)
    miss = [e for e in expected if e not in set(full)]
    rng = f"{full[0]} to {full[-1]}" if full else "none"
    partial = sorted(e for e in d if 0 < len(d[e]) < 50 and e in expected)
    detail = f" {miss}" if 0 < len(miss) <= 12 else ""
    return (f"  {name:22s} {len(full):4d} of {len(expected)} epochs with 50 seeds ({rng})"
            + ("" if not miss else f", missing {len(miss)}{detail}")
            + (f", {len(partial)} epoch(s) partly done" if partial else ""))


print("== C2ST ResNet-50 train mode, 1,500 points")
d, _ = seeds_per_epoch(f"{B}/bntrain_c2st/c2st_r50_bntrain_b*_c*.csv", stat="c2st")
print(line("r50_bntrain", d, list(range(0, 201))))

print("== MMDAgg 1,500 train mode, stored features")
for tag, exp in EXPECT.items():
    d, _ = seeds_per_epoch(f"{B}/bntrain_mmdagg/mmdagg_{tag}_b1500*_c*.csv", stat="mmdagg")
    print(line(tag, d, list(exp)))

print("== MMDAgg 1,500 train mode, SimCLR (every-5th grid, then gaps and tails)")
for ds in ("cifar10", "cifar100"):
    d, vals = seeds_per_epoch(f"{B}/ssl_bntrain/ssl_{ds}_bntrain_*_c*.csv", model=f"{ds}_bntrain", stat="mmdagg")
    grid = list(range(5, 1001, 5))
    print(line(f"{ds}_bntrain", d, grid))
    full = sorted(e for e in d if len(d[e]) >= 50)
    rej = {e: sum(v >= 0.5 for v in vals[e].values()) / len(vals[e]) >= 0.6 for e in full}
    onset = next((e for i, e in enumerate(full) if all(rej[x] for x in full[i:])), None)
    if onset is None:
        print(f"  {'':22s} no sustained onset on the evaluated epochs so far"); continue
    before = [e for e in full if e < onset]
    gap = [e for e in range(before[-1] + 1 if before else 1, onset) if e not in set(full)]
    tail = [e for e in range(onset + 1, 1001) if e not in set(full)]
    print(f"  {'':22s} sustained onset on the evaluated epochs: {onset}; gap before it {len(gap)} epoch(s)"
          + (f" ({gap[0]} to {gap[-1]})" if gap else "") + f"; after it {1000 - onset - len(tail)} of {1000 - onset} checkpoints evaluated"
          + (f", {len(tail)} to go ({tail[0]} to {tail[-1]})" if tail else "") + ("; FINAL" if not gap and not tail else "; provisional"))
