"""harvest_trtr_trained.py — the train-vs-train null on the trained ImageNet checkpoints: for each scheme, method and
budget, seeds rejecting out of 10 at the nine checkpoints under eval-mode and train-mode BatchNorm, and the sustained
onset under the paper rule (median p < 0.05 from a checkpoint on; C2ST also median accuracy > 0.55; MMDAgg at least
60 percent of seeds). Both samples are training images seen in training, so every onset should be '--' and the counts
near 5 percent of the seeds. The last column n gives the seeds per checkpoint (10 or 50).
Usage: python harvest_trtr_trained.py [root]   (default benchmark_final/s50/trtr_trained)
"""
import sys, glob, csv, collections
import numpy as np

BF = sys.argv[1] if len(sys.argv) > 1 else "/home3/p318770/venvs/overfitting/benchmark_final/s50/trtr_trained"
EPS = [1, 5, 20, 50, 100, 150, 200, 250, 300]


def rows(pattern, pcol, acccol=None, filt=None):
    out = collections.defaultdict(dict)
    for f in glob.glob(pattern):
        for r in csv.DictReader(open(f)):
            if filt and not filt(r):
                continue
            out[int(r["epoch"])][int(r["seed"])] = (float(r[pcol]), float(r[acccol]) if acccol else float("nan"))
    return out


def rej_seed(stat, p, a):
    return p >= 0.5 if stat == "mmdagg" else (p < 0.05 and a > 0.55) if stat == "c2st" else p < 0.05


def rej_epoch(stat, vals):
    ps = np.array([p for p, _ in vals]); accs = np.array([a for _, a in vals])
    if stat == "mmdagg":
        return ps.mean() >= 0.6
    if stat == "c2st":
        return np.median(ps) < 0.05 and np.median(accs) > 0.55
    return np.median(ps) < 0.05


METHODS = [
    ("VRT fwd", "10000", "p", lambda t: rows(f"{BF}/vrt_fwd/vrt_fwd_{t}_large.csv", "p")),
    ("VRT fwd k15", "1500", "p", lambda t: rows(f"{BF}/vrt_fwd/vrt_fwd_{t}_small.csv", "p")),
    ("VRT rev", "10000", "p", lambda t: rows(f"{BF}/reversed/vrt_rev_{t}_large.csv", "p")),
    ("VRT rev k15", "1500", "p", lambda t: rows(f"{BF}/reversed/vrt_rev_{t}_small.csv", "p")),
    ("MMD", "10000", "p", lambda t: rows(f"{BF}/mmd_p2000/mmd_{t}_n10000.csv", "p")),
    ("MMD", "1500", "p", lambda t: rows(f"{BF}/mmd_p2000/mmd_{t}_n1500.csv", "p")),
    ("ED", "10000", "p", lambda t: rows(f"{BF}/ed_p2000/ed_{t}_n10000.csv", "p")),
    ("ED", "1500", "p", lambda t: rows(f"{BF}/ed_p2000/ed_{t}_n1500.csv", "p")),
    ("MMDAgg", "5000", "mmdagg", lambda t: rows(f"{BF}/mmdagg/mmdagg_{t}_b5000*.csv", "p")),
    ("MMDAgg", "1500", "mmdagg", lambda t: rows(f"{BF}/mmdagg/mmdagg_{t}_b1500*.csv", "p")),
    ("Henze", "10000", "p", lambda t: rows(f"{BF}/knn_v2/knn_{t}_n10000_part*.csv", "p_henze")),
    ("Henze", "1500", "p", lambda t: rows(f"{BF}/knn_v2/knn_{t}_n1500_part*.csv", "p_henze")),
    ("Schilling", "10000", "p", lambda t: rows(f"{BF}/knn_v2/knn_{t}_n10000_part*.csv", "p_schilling")),
    ("C2ST", "10000", "c2st", lambda t: rows(f"{BF}/c2st/c2st_{t}_b*.csv", "p", "acc", filt=lambda r: r["budget"] == "b10000")),
    ("C2ST", "1500", "c2st", lambda t: rows(f"{BF}/c2st/c2st_{t}_b*.csv", "p", "acc", filt=lambda r: r["budget"] == "b1500")),
]

for scheme, base in (("simple", "imgnet_simple"), ("adv", "imgnet_adv")):
    print(f"=== DeepResNet50 {scheme}, 50,000 training images vs 10,000 further training images (null holds), seeds rejecting at checkpoints {' '.join(map(str, EPS))} ===")
    print(f"{'method':12s} {'budget':>6s} {'BN':>5s} " + " ".join(f"{e:>4d}" for e in EPS) + "  onset    n")
    for name, N, stat, src in METHODS:
        for mode, tag in (("eval", f"{base}_trtr"), ("train", f"{base}_bntrain_trtr")):
            d = src(tag)
            nseed = max((len(v) for v in d.values()), default=0)
            if not d:
                cells = ["   ." for _ in EPS]; onset = "."
            else:
                cells = [f"{sum(rej_seed(stat, p, a) for p, a in d[e].values()):4d}" if e in d else "   ." for e in EPS]
                have = [e for e in EPS if e in d]
                R = {e: rej_epoch(stat, list(d[e].values())) for e in have}
                onset = next((str(e) for i, e in enumerate(have) if all(R[x] for x in have[i:])), "--")
            print(f"{name if mode == 'eval' else '':12s} {N if mode == 'eval' else '':>6s} {mode:>5s} " + " ".join(cells) + f"  {onset:>5s}  {nseed:>3d}")
    print()
