"""harvest_s50.py — onset tables for the 50-seed rerun. Usage:
    python harvest_s50.py [s50 root] [--first] [--seeds 50]
Decision per epoch over the seeds: median p < 0.05 (C2ST additionally median held-out accuracy > 0.55;
MMDAgg at least 60 percent of seeds rejecting, the 6-of-10 rule scaled). Sustained onset = first evaluated
epoch from which every later evaluated epoch rejects. "--" no sustained detection, "." not computed.
A cell followed by [k] has only k seeds on some epoch, one followed by (e/E) was evaluated on e of the
E epochs the most complete cell of that setting has (chunks still running, or a sparser grid); --first prints
first-rejection/onset pairs instead of onsets alone. Tables: large tier (10,000 per split, MMDAgg 5,000),
small tier (1,500), raw-feature MMD/ED, BatchNorm eval -> train for the _bntrain sets, and a coverage list.
"""
import sys, glob, os, csv, collections, functools
import numpy as np

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("root", nargs="?", default="/home3/p318770/venvs/overfitting/benchmark_final/s50")
_ap.add_argument("--first", action="store_true", help="print first-rejection/onset pairs")
_ap.add_argument("--seeds", type=int, default=50, help="seeds expected per cell (coverage flag)")
_ap.add_argument("--feats", default="/scratch/p318770/model_zoos/ladder_feats", help="feature root; its ep*.npz counts are the reference for the (e/E) completeness flag")
_a = _ap.parse_args()
BF, FIRST, NS, FEATS = _a.root, _a.first, _a.seeds, _a.feats

EVAL = (["r50", "phase_bs8", "phase_bs128", "phase_bs1024", "v7", "ablA", "ablB", "ablC", "ablD", "wp"]
        + [f"zoo_c10_m{i}" for i in range(5)] + [f"zoo_c100_m{i}" for i in range(5)] + ["imgnet_simple", "imgnet_adv"])
TRAIN = ([f"{t}_bntrain" for t in ["r50", "ablA", "ablB", "ablC", "ablD"]]
         + [f"zoo_c10_m{i}_bntrain" for i in range(5)] + [f"zoo_c100_m{i}_bntrain" for i in range(5)]
         + ["imgnet_simple_bntrain", "imgnet_adv_bntrain"])
SSL = ["simclr_c10", "simclr_c100"]


@functools.lru_cache(maxsize=None)
def read(pattern, pcol, acccol=None, epcol="epoch", seedcol="seed", filt=None):
    """epoch -> {seed: (p, acc)} over all files matching the pattern (cached; filt must be hashable)."""
    d = collections.defaultdict(dict)
    for f in glob.glob(pattern):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if filt and not filt(r):
                    continue
                acc = float(r[acccol]) if acccol else float("nan")
                d[int(r[epcol])][int(r[seedcol])] = (float(r[pcol]), acc)
    return d


class Filt:
    """hashable row filter: every given column must equal the given value"""
    def __init__(self, **kw): self.kw = tuple(sorted(kw.items()))
    def __call__(self, r): return all(r[k] == v for k, v in self.kw)
    def __hash__(self): return hash(self.kw)
    def __eq__(self, o): return isinstance(o, Filt) and self.kw == o.kw


def cell_data(tag, stat, N):
    """stat in vrt, vrt_rev, mmd, ed, mmd_raw, ed_raw, henze, schilling, c2st, mmdagg; N = 1500 or 10000 (5000 for mmdagg large)."""
    if tag in SSL:
        ds = "cifar10" if tag == "simclr_c10" else "cifar100"
        if stat == "mmdagg" and N != 1500:
            return None                                        # MMDAgg 5,000 for SimCLR: ssl_mmdagg5k_seeds.py, own harvest
        # vrt_rev, henze, schilling, mmd_raw, ed_raw come from the rev / knn / extra / raw groups of ssl_s50_fill.py
        # (October 8; rev and knn on an every-5-epoch grid), same files, same columns
        st = {"vrt": "vrt" if N == 10000 else "vrt_low"}.get(stat, stat)
        return read(f"{BF}/ssl/ssl_{ds}_*_c*.csv", "p", "acc", filt=Filt(model=ds, stat=st, budget=f"b{N}"))
    b = "large" if N == 10000 else "small"
    if stat == "vrt":
        return read(f"{BF}/vrt_fwd/vrt_fwd_{tag}_{b}_c*.csv", "p")
    if stat == "vrt_rev":
        return read(f"{BF}/reversed/vrt_rev_{tag}_{b}_c*.csv", "p")
    if stat == "mmd":
        return read(f"{BF}/mmd_p2000/mmd_{tag}_n{N}_c*.csv", "p")
    if stat == "mmd_raw":
        return read(f"{BF}/mmd_raw_p2000/mmd_{tag}_n{N}_c*.csv", "p")
    if stat == "ed":
        return read(f"{BF}/ed_p2000/ed_{tag}_n{N}_c*.csv", "p")
    if stat == "ed_raw":
        return read(f"{BF}/ed_raw_p2000/ed_{tag}_n{N}_c*.csv", "p")
    if stat == "henze":
        return read(f"{BF}/knn_v2/knn_{tag}_n{N}_part*.csv", "p_henze")
    if stat == "schilling":
        return read(f"{BF}/knn_v2/knn_{tag}_n{N}_part*.csv", "p_schilling")
    if stat == "c2st":   # base_fill names the file after all budgets of the call (b1500-10000); filter on the columns
        return read(f"{BF}/c2st/c2st_{tag}_b*_c*.csv", "p", "acc", filt=Filt(model=tag, budget=f"b{N}"))
    if stat == "mmdagg":
        return read(f"{BF}/mmdagg/mmdagg_{tag}_b{N}_c*.csv", "p")
    raise ValueError(stat)


def rejects(stat, vals):
    ps = np.array([p for p, _ in vals]); accs = np.array([a for _, a in vals])
    if stat == "mmdagg":
        return ps.mean() >= 0.6
    if stat == "c2st":
        return np.median(ps) < 0.05 and np.median(accs) > 0.55
    return np.median(ps) < 0.05


def summarize(stat, d):
    """-> (first, onset, min_seeds, n_epochs) or None"""
    if not d:
        return None
    eps = sorted(d)
    rej = {e: rejects(stat, list(d[e].values())) for e in eps}
    first = next((e for e in eps if rej[e]), None)
    onset = next((e for i, e in enumerate(eps) if all(rej[x] for x in eps[i:])), None)
    return first, onset, min(len(v) for v in d.values()), len(eps)


def fmt(stat, d, n_ref=None):
    """onset text; [k] = only k seeds on some epoch, (e/E) = evaluated on e of the E epochs the most complete
    cell of this setting has (chunks still running or a sparser grid)"""
    s = summarize(stat, d)
    if s is None:
        return "."
    first, onset, mins, neps = s
    f = lambda x: "--" if x is None else str(x)
    txt = f"{f(first)}/{f(onset)}" if FIRST else f(onset)
    if mins < NS:
        txt += f"[{mins}]"
    if n_ref and neps < n_ref and stat not in ("mmdagg",):        # MMDAgg 5,000 runs on a 5-epoch grid by design
        txt += f"({neps}/{n_ref})"
    return txt


@functools.lru_cache(maxsize=None)
def n_epochs_ref(tag, N):
    """reference epoch count for the (e/E) completeness flag: the number of saved feature files of the setting
    (ep*.npz in the feature folder; 1,000 for SimCLR); falls back to the most complete cell when the folder is absent"""
    if tag in SSL:
        return 1000
    fs = glob.glob(os.path.join(FEATS, tag, "ep*.npz"))
    if fs:
        return len(fs)
    best = 0
    for stat in ("vrt", "mmd", "ed", "henze", "c2st", "mmd_raw", "ed_raw", "vrt_rev"):
        d = cell_data(tag, stat, N)
        if d:
            best = max(best, len(d))
    return best


def table(title, rows, cols, N_of):
    print(f"=== {title} ===")
    w = 16 if FIRST else 12
    print(f"{'setting':22s} " + " ".join(f"{c:>{w}s}" for c, _ in cols))
    for tag in rows:
        cells = []
        for _, stat in cols:
            N = N_of(stat)
            d = cell_data(tag, stat, N)
            cells.append("n/a" if d is None else fmt(stat, d, n_epochs_ref(tag, 10000 if N == 5000 else N)))
        print(f"{tag:22s} " + " ".join(f"{c:>{w}s}" for c in cells))
    print()


LARGE = [("VRT", "vrt"), ("MMD", "mmd"), ("MMDAgg5k", "mmdagg"), ("C2ST", "c2st"), ("ED", "ed"),
         ("Henze", "henze"), ("Schilling", "schilling"), ("VRT rev", "vrt_rev")]
SMALL = [("VRT k15", "vrt"), ("MMD", "mmd"), ("MMDAgg", "mmdagg"), ("C2ST", "c2st"), ("ED", "ed"),
         ("Henze", "henze"), ("Schilling", "schilling"), ("VRT rev", "vrt_rev")]
table(f"large tier, 10,000 per split (MMDAgg 5,000), {NS} seeds", EVAL + SSL, LARGE, lambda s: 5000 if s == "mmdagg" else 10000)
table(f"small tier, 1,500 per split, {NS} seeds", EVAL + SSL, SMALL, lambda s: 1500)
table("raw (un-normalized) features, 10,000 per split", EVAL, [("MMD raw", "mmd_raw"), ("ED raw", "ed_raw"), ("MMD norm", "mmd"), ("ED norm", "ed")], lambda s: 10000)
table("raw (un-normalized) features, 1,500 per split", EVAL, [("MMD raw", "mmd_raw"), ("ED raw", "ed_raw"), ("MMD norm", "mmd"), ("ED norm", "ed")], lambda s: 1500)

# BatchNorm eval -> train
for N in (10000, 1500):
    print(f"=== BatchNorm eval-mode -> train-mode features, {N} per split ===")
    cols = [("VRT", "vrt"), ("MMD", "mmd"), ("ED", "ed"), ("Henze", "henze"), ("MMD raw", "mmd_raw"), ("ED raw", "ed_raw")]
    print(f"{'setting':22s} " + " ".join(f"{c:>20s}" for c, _ in cols))
    for t in TRAIN:
        base = t[:-len("_bntrain")]
        cells = [f"{fmt(stat, cell_data(base, stat, N), n_epochs_ref(base, N))} -> {fmt(stat, cell_data(t, stat, N), n_epochs_ref(t, N))}" for _, stat in cols]
        print(f"{base:22s} " + " ".join(f"{c:>20s}" for c in cells))
    print()

# ImageNet: every method, eval-mode -> train-mode, both schemes (MMDAgg at 5,000 in the 10,000 tier)
for N in (10000, 1500):
    print(f"=== ImageNet eval-mode -> train-mode, {N} per split{' (MMDAgg 5,000)' if N == 10000 else ''} ===")
    meths = [("VRT", "vrt"), ("MMD", "mmd"), ("MMDAgg", "mmdagg"), ("C2ST", "c2st"), ("ED", "ed"),
             ("Henze", "henze"), ("Schilling", "schilling"), ("MMD raw", "mmd_raw"), ("ED raw", "ed_raw")]
    print(f"{'method':10s} {'simple':>24s} {'adv':>24s}")
    for name, stat in meths:
        cells = []
        for sch in ("imgnet_simple", "imgnet_adv"):
            NN = 5000 if (stat == "mmdagg" and N == 10000) else N
            e = fmt(stat, cell_data(sch, stat, NN), n_epochs_ref(sch, N))
            t = fmt(stat, cell_data(sch + "_bntrain", stat, NN), n_epochs_ref(sch + "_bntrain", N))
            cells.append(f"{e} -> {t}")
        print(f"{name:10s} " + " ".join(f"{c:>24s}" for c in cells))
    print()

# coverage: cells with incomplete seeds
inc = []
for tag in EVAL + TRAIN + SSL:
    for N in (1500, 10000):
        for stat in ("vrt", "vrt_rev", "mmd", "ed", "mmd_raw", "ed_raw", "henze", "c2st", "mmdagg"):
            NN = 5000 if (stat == "mmdagg" and N == 10000) else N
            d = cell_data(tag, stat, NN)
            if d:
                s = summarize(stat, d)
                ref = n_epochs_ref(tag, N)
                if s[2] < NS or (s[3] < ref and stat != "mmdagg"):
                    inc.append(f"{tag} {stat} n{NN}: {s[3]} of {ref} epochs, min seeds {s[2]}")
print(f"incomplete cells (fewer than {NS} seeds on some epoch, or fewer epochs than the setting's most complete cell): {len(inc)}")
for line in inc[:60]:
    print("  " + line)
