"""
dump_ssl_modes.py — SimCLR per-epoch results under eval-mode and train-mode BatchNorm extraction, side by side, for the
check of the Table 2 SimCLR row (camera-ready #27102).

Reads the s50 fill files (columns model,epoch,budget,stat,seed,acc,p):
  eval mode    $S50/ssl/ssl_<ds>_*_c*.csv
  train mode   $S50/ssl_bntrain/ssl_<ds>_bntrain_*_c*.csv
For every (stat, budget) it prints the coverage (epochs with all 50 seeds), the first epoch with median p < 0.05 and the
sustained onset so far (paper rule: median p over the seeds below 0.05, for C2ST also median held-out accuracy above 0.55,
from the first complete epoch after which every later complete epoch rejects; "so far" because the train-mode run is not
finished), then one row per epoch with, for each mode, the median p, the number of seeds with p < 0.05 and (C2ST) the
median accuracy. Rows: every epoch that is complete in train mode, plus every --every-th epoch of the eval-mode run.

Duplicate rows (same stat, budget, epoch and seed in two files, possible when two task groups overlap) are counted once.
Usage:  S50=/home3/p318770/venvs/overfitting/benchmark_final/s50 python3 dump_ssl_modes.py --ds cifar10 > ssl_c10_modes.txt
        python3 dump_ssl_modes.py --ds cifar10 --every 5 --lo 100 --hi 500 --stats vrt,mmd,ed,c2st
"""
import argparse
import collections
import csv
import glob
import os
import statistics as st

S50 = os.environ.get("S50", "/home3/p318770/venvs/overfitting/benchmark_final/s50")
N_SEEDS = 50


def load(pattern):
    seen = {}                                   # (stat, budget, epoch, seed) -> (p, acc); first occurrence wins
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                try:
                    k = (r["stat"], r["budget"], int(r["epoch"]), int(r["seed"]))
                    if k not in seen:
                        seen[k] = (float(r["p"]), float(r["acc"]))
                except (KeyError, ValueError):
                    continue
    by = collections.defaultdict(list)          # (stat, budget, epoch) -> [(p, acc)], one entry per seed
    for (s, b, e, _), v in seen.items():
        by[(s, b, e)].append(v)
    return by


def rejects(stat, vals):
    if stat == "mmdagg":                       # the p column holds the 0/1 reject flag; paper rule: at least 60% of the seeds reject
        return sum(p for p, _ in vals) / len(vals) >= 0.6
    ps = [p for p, _ in vals]
    ok = st.median(ps) < 0.05
    if stat.startswith("c2st"):
        ok = ok and st.median(a for _, a in vals) > 0.55
    return ok


def ranges(xs):
    """compact text of a sorted set of epochs: consecutive runs as a-b, arithmetic runs (every k epochs) as a-b/k"""
    xs = sorted(set(xs))
    if not xs:
        return "-"
    out, i = [], 0
    while i < len(xs):
        j = i
        step = xs[i + 1] - xs[i] if i + 1 < len(xs) else 1
        while j + 1 < len(xs) and xs[j + 1] - xs[j] == step:
            j += 1
        if j - i >= 2 and step > 1:
            out.append("%d-%d/%d" % (xs[i], xs[j], step)); i = j + 1
        elif j - i >= 1 and step == 1:
            out.append("%d-%d" % (xs[i], xs[j])); i = j + 1
        else:
            out.append("%d" % xs[i]); i += 1
    return " ".join(out)


def onset(stat, by, key):
    eps = sorted(e for (s, b, e) in by if (s, b) == key and len(by[(s, b, e)]) >= N_SEEDS)
    rej = {e for e in eps if rejects(stat, by[key + (e,)])}
    first = min(rej) if rej else None
    sust = next((e for i, e in enumerate(eps) if all(x in rej for x in eps[i:])), None)
    return eps, rej, first, sust


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--every", type=int, default=5, help="eval-mode rows at every k-th epoch (train-mode rows: all)")
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--hi", type=int, default=1000)
    ap.add_argument("--stats", default="", help="comma list, e.g. vrt,mmd,ed,c2st (default: all)")
    args = ap.parse_args()
    ev = load(os.path.join(S50, "ssl", "ssl_%s_*_c*.csv" % args.ds))
    tr = load(os.path.join(S50, "ssl_bntrain", "ssl_%s_bntrain_*_c*.csv" % args.ds))
    keys = sorted({(s, b) for (s, b, _) in list(ev) + list(tr)}, key=lambda k: (k[0], -int(k[1][1:])))
    if args.stats:
        want = set(args.stats.split(","))
        keys = [k for k in keys if k[0] in want]
    print("# SimCLR %s, eval-mode versus train-mode BatchNorm extraction; 50 seeds per epoch; rule = median p < 0.05"
          " (C2ST also median accuracy > 0.55; MMDAgg at least 60%% of the seeds); sustained onset = first complete epoch after which every later complete"
          " epoch rejects" % args.ds)
    print("# eval rows every %d epochs, train rows every complete epoch, epochs %d to %d" % (args.every, args.lo, args.hi))
    for stat, budget in keys:
        key = (stat, budget)
        eps_e, rej_e, first_e, sust_e = onset(stat, ev, key)
        eps_t, rej_t, first_t, sust_t = onset(stat, tr, key)
        print("\n== %s %s" % (stat, budget))
        print("   eval : complete %s | first median p<0.05 at %s | sustained onset %s | rejecting epochs %s"
              % (ranges(eps_e), first_e, sust_e, ranges(sorted(rej_e))))
        print("   train: complete %s | first median p<0.05 at %s | sustained onset so far %s | rejecting epochs %s"
              % (ranges(eps_t), first_t, sust_t, ranges(sorted(rej_t))))
        head = "   epoch | eval  med p  n<.05" + ("  med acc" if stat.startswith("c2st") else "") + \
               " | train med p  n<.05" + ("  med acc" if stat.startswith("c2st") else "")
        print(head)
        rows = sorted(set(e for e in eps_t if args.lo <= e <= args.hi)
                      | set(e for e in eps_e if args.lo <= e <= args.hi and e % args.every == 0))
        for e in rows:
            cells = []
            for by, eps in ((ev, eps_e), (tr, eps_t)):
                if e in eps:
                    vals = by[key + (e,)]
                    ps = [p for p, _ in vals]
                    c = "%6.3f  %2d/%d" % (st.median(ps), sum(p < 0.05 for p in ps), len(ps))
                    if stat.startswith("c2st"):
                        c += "  %5.3f" % st.median(a for _, a in vals)
                    cells.append(c)
                else:
                    cells.append("     -      -" + ("      -" if stat.startswith("c2st") else ""))
            print("   %5d | %s | %s" % (e, cells[0], cells[1]))


if __name__ == "__main__":
    main()
