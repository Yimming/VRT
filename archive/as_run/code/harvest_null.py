"""harvest_null.py — rejection rates on the random-initialization ImageNet feature sets for every method, one row per
method and budget, eval-mode and train-mode BatchNorm side by side. Each init (= 'epoch' index) may hold one or
several analysis seeds:
   a/N      inits whose decision under the paper rule rejects (median p over the seeds < 0.05; C2ST also median
            held-out accuracy > 0.55; MMDAgg at least 60 percent of the seeds rejecting)
   (b%)     percentage of all (init, seed) tests rejecting at alpha 0.05, i.e. the rate averaged over reference
            and subsample draws
With one seed per init the two numbers coincide. Usage:
  python harvest_null.py [root] [tag suffix]
  python harvest_null.py /home3/p318770/venvs/overfitting/benchmark_final/s50/null10            train vs validation
  python harvest_null.py /home3/p318770/venvs/overfitting/benchmark_final/s50/null10 _trtr      train vs further train
"""
import sys, glob, csv, collections
import numpy as np

BF = sys.argv[1] if len(sys.argv) > 1 else "/home3/p318770/venvs/overfitting/benchmark_final/s50/null"
SUF = sys.argv[2] if len(sys.argv) > 2 else ""          # "_trtr" for the train-vs-train features


def rows(pattern, pcol, acccol=None, filt=None):
    """-> {init: {seed: (p, acc)}}"""
    out = collections.defaultdict(dict)
    for f in glob.glob(pattern):
        for r in csv.DictReader(open(f)):
            if filt and not filt(r):
                continue
            out[int(r["epoch"])][int(r["seed"])] = (float(r[pcol]), float(r[acccol]) if acccol else None)
    return out


def decide(stat, vals):
    ps = np.array([p for p, _ in vals]); accs = np.array([a for _, a in vals], dtype=float)
    if stat == "mmdagg":
        return ps.mean() >= 0.6
    if stat == "c2st":
        return np.median(ps) < 0.05 and np.median(accs) > 0.55
    return np.median(ps) < 0.05


def single(stat, p, a):
    if stat == "mmdagg":
        return p >= 0.5
    if stat == "c2st":
        return p < 0.05 and a > 0.55
    return p < 0.05


def rate(d, stat):
    if not d:
        return "        ."
    n = len(d)
    k = sum(decide(stat, list(v.values())) for v in d.values())
    tests = [(p, a) for v in d.values() for p, a in v.values()]
    pct = 100.0 * sum(single(stat, p, a) for p, a in tests) / len(tests)
    nseed = sorted({len(v) for v in d.values()})
    return f"{k:2d}/{n:<2d} ({pct:4.1f}%)" + ("" if nseed == [1] else f" s{nseed[0]}" if len(nseed) == 1 else f" s{nseed[0]}-{nseed[-1]}")


print(f"{'method':22s} {'budget':>6s} {'eval-mode BN':>22s} {'train-mode BN':>22s}")
for name, N, src in [
    ("VRT forward", "large", lambda t: rows(f"{BF}/vrt_fwd/vrt_fwd_{t}_large.csv", "p")),
    ("VRT forward k15", "small", lambda t: rows(f"{BF}/vrt_fwd/vrt_fwd_{t}_small.csv", "p")),
    ("VRT reversed", "large", lambda t: rows(f"{BF}/reversed/vrt_rev_{t}_large.csv", "p")),
    ("VRT reversed k15", "small", lambda t: rows(f"{BF}/reversed/vrt_rev_{t}_small.csv", "p")),
    ("MMD", "10000", lambda t: rows(f"{BF}/mmd_p2000/mmd_{t}_n10000.csv", "p")),
    ("MMD", "1500", lambda t: rows(f"{BF}/mmd_p2000/mmd_{t}_n1500.csv", "p")),
    ("MMD raw", "10000", lambda t: rows(f"{BF}/mmd_raw_p2000/mmd_{t}_n10000.csv", "p")),
    ("ED", "10000", lambda t: rows(f"{BF}/ed_p2000/ed_{t}_n10000.csv", "p")),
    ("ED", "1500", lambda t: rows(f"{BF}/ed_p2000/ed_{t}_n1500.csv", "p")),
    ("ED raw", "10000", lambda t: rows(f"{BF}/ed_raw_p2000/ed_{t}_n10000.csv", "p")),
    ("Henze", "10000", lambda t: rows(f"{BF}/knn_v2/knn_{t}_n10000_part*.csv", "p_henze")),
    ("Henze", "1500", lambda t: rows(f"{BF}/knn_v2/knn_{t}_n1500_part*.csv", "p_henze")),
    ("Schilling", "10000", lambda t: rows(f"{BF}/knn_v2/knn_{t}_n10000_part*.csv", "p_schilling")),
    ("MMDAgg", "5000", lambda t: rows(f"{BF}/mmdagg/mmdagg_{t}_b5000*.csv", "p")),
    ("MMDAgg", "1500", lambda t: rows(f"{BF}/mmdagg/mmdagg_{t}_b1500*.csv", "p")),
    ("C2ST", "10000", lambda t: rows(f"{BF}/c2st/c2st_{t}_b*.csv", "p", "acc", filt=lambda r: r["budget"] == "b10000")),
    ("C2ST", "1500", lambda t: rows(f"{BF}/c2st/c2st_{t}_b*.csv", "p", "acc", filt=lambda r: r["budget"] == "b1500")),
]:
    stat = "mmdagg" if name == "MMDAgg" else ("c2st" if name == "C2ST" else "p")
    print(f"{name:22s} {N:>6s} {rate(src('imgnet_rand_eval' + SUF), stat):>22s} {rate(src('imgnet_rand_train' + SUF), stat):>22s}")
print("a/N = inits rejecting under the paper's median-over-seeds rule, (b%) = share of all (init, seed) tests rejecting, sK = seeds per init;")
print("50,000 train vs 10,000 FURTHER TRAIN images (the two-sample null holds): a calibrated test stays near 5 percent" if SUF
      else "50,000 train vs 10,000 VALIDATION images (two ImageNet splits, untrained network): rejections measure the input-level split difference")
