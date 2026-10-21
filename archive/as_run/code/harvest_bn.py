"""harvest_bn.py — Section 4.4 check on the camera-ready ImageNet features: every method under eval-mode
BatchNorm (running statistics, the Table 1 protocol) next to the same method on the train-mode-BatchNorm
features (per-batch statistics), same images, same seeds, same protocol.
Usage: python harvest_bn.py [benchmark_final dir]
Cell = number of seeds (of 10) rejecting at that checkpoint (p < 0.05; MMDAgg reject flag; C2ST additionally
counts only seeds with held-out accuracy > 0.55). Onset = first checkpoint from which the group rule holds at
every later checkpoint (median p < 0.05 and median acc > 0.55 for C2ST; at least 6 of 10 MMDAgg rejections).
"""
import sys, glob, os, csv, collections
import numpy as np

BF = sys.argv[1] if len(sys.argv) > 1 else "/home3/p318770/venvs/overfitting/benchmark_final"
EPOCHS = [1, 5, 20, 50, 100, 150, 200, 250, 300]


def read(pattern, pcol, epcol="epoch", seedcol="seed", acccol=None, tagcol=None, tag=None):
    """epoch -> {seed: (p, acc)} from every file matching the pattern."""
    d = collections.defaultdict(dict)
    for f in glob.glob(pattern):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if tagcol and tag and r[tagcol] != tag:
                    continue
                acc = float(r[acccol]) if acccol else np.nan
                d[int(r[epcol])][int(r[seedcol])] = (float(r[pcol]), acc)
    return d


def sources(tag):
    """(method, budget) -> epoch -> {seed: (p, acc)}"""
    S = {}
    for N in ("10000", "1500"):
        S[("MMD", N)] = read(f"{BF}/mmd_p2000/mmd_{tag}_n{N}.csv", "p")
        S[("ED", N)] = read(f"{BF}/ed_p2000/ed_{tag}_n{N}.csv", "p")
        S[("Henze", N)] = read(f"{BF}/knn_v2/knn_{tag}_n{N}_part*.csv", "p_henze")
        S[("Schilling", N)] = read(f"{BF}/knn_v2/knn_{tag}_n{N}_part*.csv", "p_schilling")
    S[("MMDAgg", "5000")] = read(f"{BF}/mmdagg_10seed/mmdagg_{tag}_b5000*.csv", "p")
    S[("MMDAgg", "1500")] = read(f"{BF}/mmdagg_10seed/mmdagg_{tag}_b1500*.csv", "p")
    c2 = collections.defaultdict(lambda: collections.defaultdict(dict))
    for f in glob.glob(f"{BF}/c2st_10seed/c2st_{tag}_b*.csv"):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r["model"] != tag:      # c2st_imgnet_simple_b* also matches the _bntrain files
                    continue
                c2[r["budget"][1:]][int(r["epoch"])][int(r["seed"])] = (float(r["p"]), float(r["acc"]))
    for N in ("10000", "1500"):
        S[("C2ST", N)] = c2.get(N, {})
    S[("VRT fwd", "10000")] = read(f"{BF}/vrt_fwd/vrt_fwd_{tag}_large.csv", "p")
    S[("VRT fwd", "1500")] = read(f"{BF}/ksweep15_extra/ksweep15_{tag}_*.csv", "p", tagcol="model", tag=tag)
    return S


def rejects(method, vals):
    if method == "MMDAgg":
        return sum(int(p) for p, _ in vals)
    if method == "C2ST":
        return sum(1 for p, a in vals if p < 0.05 and a > 0.55)
    return sum(1 for p, _ in vals if p < 0.05)


def group_reject(method, vals):
    ps = np.array([p for p, _ in vals]); accs = np.array([a for _, a in vals])
    if method == "MMDAgg":
        return ps.mean() >= 0.6
    if method == "C2ST":
        return np.median(ps) < 0.05 and np.median(accs) > 0.55
    return np.median(ps) < 0.05


def onset(method, d):
    eps = sorted(d)
    rej = {e: group_reject(method, list(d[e].values())) for e in eps}
    for i, e in enumerate(eps):
        if all(rej[x] for x in eps[i:]):
            return e
    return None


ROWS = [("VRT fwd", "10000"), ("MMD", "10000"), ("ED", "10000"), ("MMDAgg", "5000"), ("Henze", "10000"),
        ("Schilling", "10000"), ("C2ST", "10000"),
        ("VRT fwd", "1500"), ("MMD", "1500"), ("ED", "1500"), ("MMDAgg", "1500"), ("Henze", "1500"),
        ("Schilling", "1500"), ("C2ST", "1500")]

for scheme in ("simple", "adv"):
    base = f"imgnet_{scheme}"
    S = {"eval": sources(base), "train": sources(base + "_bntrain")}
    if not any(S[m][k] for m in S for k in S[m]):
        print(f"=== DeepResNet50 {scheme}: no results yet ===\n"); continue
    print(f"=== DeepResNet50 {scheme}: seeds rejecting out of 10 at checkpoints {' '.join(map(str, EPOCHS))}; eval = running statistics (Table 1), train = per-batch BatchNorm statistics ===")
    print(f"{'method':10s} {'budget':>6s} {'BN':>5s} " + " ".join(f"{e:>4d}" for e in EPOCHS) + f" {'onset':>6s}")
    for method, N in ROWS:
        for mode in ("eval", "train"):
            d = S[mode].get((method, N), {})
            if not d:
                cells = " ".join(f"{'.':>4s}" for _ in EPOCHS); on = "n/a"
            else:
                cells = " ".join(f"{rejects(method, list(d[e].values())):4d}" if e in d else f"{'.':>4s}" for e in EPOCHS)
                o = onset(method, d); on = "--" if o is None else str(o)
                nseeds = min(len(v) for v in d.values())
                if nseeds < 10:
                    on += f" [{nseeds}s]"
            label = f"{method:10s} {N:>6s}" if mode == "eval" else " " * 17
            print(f"{label} {mode:>5s} {cells} {on:>6s}")
    print()
print("onset rule: median p < 0.05 (C2ST also median acc > 0.55; MMDAgg at least 6/10 rejections) at the checkpoint and every later one.  '.' = not computed, [ks] = only k seeds on some checkpoint")
