"""LEGACY (camera-ready code release note): consistency check between the 50-seed rerun and the published 10-seed results (transition stage), not a table entry point. Kept as run for provenance; not an entry point of the release."""
"""s50_check.py — consistency check between the 50-seed rerun and the published 10-seed results: for every
(setting, method, budget, epoch, seed) present in both, compare the p-values. Same code, same features and
same seed must give the same p (up to the decimals written), so any difference points at a protocol change.
Usage: python s50_check.py [s50 root] [old benchmark_final root]
"""
import sys, glob, os, csv, collections
import numpy as np

NEW = sys.argv[1] if len(sys.argv) > 1 else "/home3/p318770/venvs/overfitting/benchmark_final/s50"
OLD = sys.argv[2] if len(sys.argv) > 2 else "/home3/p318770/venvs/overfitting/benchmark_final"
TAGS = (["r50", "phase_bs8", "phase_bs128", "phase_bs1024", "v7", "ablA", "ablB", "ablC", "ablD", "wp"]
        + [f"zoo_c10_m{i}" for i in range(5)] + [f"zoo_c100_m{i}" for i in range(5)] + ["imgnet_simple", "imgnet_adv"])
BNT = [f"{t}_bntrain" for t in ["r50", "ablA", "ablB", "ablC", "ablD"] + [f"zoo_c10_m{i}" for i in range(5)] + [f"zoo_c100_m{i}" for i in range(5)] + ["imgnet_simple", "imgnet_adv"]]


def load(pattern, pcol, key=lambda r: (int(r["epoch"]), int(r["seed"])), filt=None, tagcol=None):
    out = {}
    for f in glob.glob(pattern):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if filt and not filt(r):
                    continue
                k = key(r)
                if tagcol:
                    k = (r[tagcol],) + k
                out[k] = float(r[pcol])
    return out


def compare(name, old, new, tol):
    common = set(old) & set(new)
    if not common:
        print(f"{name:34s} no overlap (old {len(old)}, new {len(new)})"); return
    diffs = np.array([abs(old[k] - new[k]) for k in common])
    bad = int((diffs > tol).sum()); big = int((diffs > 1e-3).sum())     # > one permutation count (1/2001)
    worst = sorted(common, key=lambda k: -abs(old[k] - new[k]))[:3]
    print(f"{name:34s} overlap {len(common):6d}  max|dp| {diffs.max():.2e}  differing {bad} (beyond one count {big})" +
          ("" if bad == 0 else "   e.g. " + "; ".join(f"{k}: {old[k]:.4f} vs {new[k]:.4f}" for k in worst)))


# where the 10-seed results of each feature set live: eval-mode sets and the ImageNet train-mode sets in the
# top-level folders (job_imgnet_bntrain_tests.sh), the CIFAR train-mode sets under bntrain/ (job_bntrain_base.sh)
def old_dir(tag, sub):
    if tag in TAGS or tag.startswith("imgnet"):
        return f"{OLD}/{sub}"
    return f"{OLD}/bntrain/{sub}"


for tag in TAGS + BNT:
    for N in (1500, 10000):
        compare(f"mmd {tag} n{N}", load(f"{old_dir(tag, 'mmd_p2000')}/mmd_{tag}_n{N}.csv", "p"), load(f"{NEW}/mmd_p2000/mmd_{tag}_n{N}_c*.csv", "p"), 1e-6)
        compare(f"ed {tag} n{N}", load(f"{old_dir(tag, 'ed_p2000')}/ed_{tag}_n{N}.csv", "p"), load(f"{NEW}/ed_p2000/ed_{tag}_n{N}_c*.csv", "p"), 1e-6)
        compare(f"mmd raw {tag} n{N}", load(f"{old_dir(tag, 'mmd_raw_p2000')}/mmd_{tag}_n{N}.csv", "p"), load(f"{NEW}/mmd_raw_p2000/mmd_{tag}_n{N}_c*.csv", "p"), 1e-6)
        compare(f"ed raw {tag} n{N}", load(f"{old_dir(tag, 'ed_raw_p2000')}/ed_{tag}_n{N}.csv", "p"), load(f"{NEW}/ed_raw_p2000/ed_{tag}_n{N}_c*.csv", "p"), 1e-6)
        compare(f"henze {tag} n{N}", load(f"{old_dir(tag, 'knn_v2')}/knn_{tag}_n{N}_part*.csv", "p_henze"), load(f"{NEW}/knn_v2/knn_{tag}_n{N}_part*.csv", "p_henze"), 1e-4)
        compare(f"schilling {tag} n{N}", load(f"{old_dir(tag, 'knn_v2')}/knn_{tag}_n{N}_part*.csv", "p_schilling"), load(f"{NEW}/knn_v2/knn_{tag}_n{N}_part*.csv", "p_schilling"), 1e-6)
    compare(f"vrt fwd {tag} large", load(f"{old_dir(tag, 'vrt_fwd')}/vrt_fwd_{tag}_large.csv", "p"), load(f"{NEW}/vrt_fwd/vrt_fwd_{tag}_large_c*.csv", "p"), 1e-6)
    ks = f"{OLD}/ksweep15*/ksweep15_{tag}_*.csv" if (tag in TAGS or tag.startswith("imgnet")) else f"{OLD}/bntrain/ksweep15/ksweep15_{tag}_*.csv"
    compare(f"vrt small (ksweep15) {tag}", load(ks, "p", filt=lambda r, t=tag: r["model"] == t), load(f"{NEW}/vrt_fwd/vrt_fwd_{tag}_small_c*.csv", "p"), 1e-6)
    if tag in TAGS:
        for b in ("large", "small"):
            compare(f"vrt rev {tag} {b}", load(f"{OLD}/reversed/vrt_rev_{tag}_{b}.csv", "p"), load(f"{NEW}/reversed/vrt_rev_{tag}_{b}_c*.csv", "p"), 1e-4)
        for N in (1500, 10000):
            compare(f"c2st {tag} b{N}", load(f"{OLD}/c2st_10seed/c2st_{tag}_b*.csv", "p", filt=lambda r, N=N, t=tag: r["budget"] == f"b{N}" and r["model"] == t),
                    load(f"{NEW}/c2st/c2st_{tag}_b*_c*.csv", "p", filt=lambda r, N=N, t=tag: r["budget"] == f"b{N}" and r["model"] == t), 1e-6)
        for N in (1500, 5000):
            compare(f"mmdagg {tag} b{N}", load(f"{OLD}/mmdagg_10seed/mmdagg_{tag}_b{N}*.csv", "p"), load(f"{NEW}/mmdagg/mmdagg_{tag}_b{N}_c*.csv", "p"), 1e-6)

for ds in ("cifar10", "cifar100"):
    new = lambda st, N: load(f"{NEW}/ssl/ssl_{ds}_*_c*.csv", "p", filt=lambda r, st=st, N=N: r["stat"] == st and r["budget"] == f"b{N}")
    compare(f"simclr {ds} vrt b10000", load(f"{OLD}/vrt_{ds}_SimCLR*.csv", "p", filt=lambda r: r.get("space", "backbone") == "backbone"), new("vrt", 10000), 1e-6)
    compare(f"simclr {ds} vrt extra seeds", load(f"{OLD}/ssl_baselines/sslseeds_{ds}_*.csv", "p", filt=lambda r: r["stat"] == "vrt"), new("vrt", 10000), 1e-6)
    compare(f"simclr {ds} vrt_low b1500", load(f"{OLD}/ssl_baselines/sslvrt_{ds}_*.csv", "p", filt=lambda r: r["stat"] == "vrt_low"), new("vrt_low", 1500), 1e-6)
    for st in ("mmd", "ed"):
        for N in (1500, 10000):
            compare(f"simclr {ds} {st} b{N}", load(f"{OLD}/ssl_baselines/sslbase_{ds}_*.csv", "p", filt=lambda r, st=st, N=N: r["stat"] == st and r["budget"] == f"b{N}"), new(st, N), 1e-6)
        compare(f"simclr {ds} {st} extra seeds", load(f"{OLD}/ssl_baselines/sslseeds_{ds}_*.csv", "p", filt=lambda r, st=st: r["stat"] == st), new(st, 10000), 1e-6)
    for N in (1500, 10000):
        compare(f"simclr {ds} c2st b{N}", load(f"{OLD}/ssl_baselines/sslbase_{ds}_*.csv", "p", filt=lambda r, N=N: r["stat"] == "c2st" and r["budget"] == f"b{N}"), new("c2st", N), 1e-6)
    compare(f"simclr {ds} mmdagg b1500", load(f"{OLD}/ssl_baselines/sslbase_{ds}_*.csv", "p", filt=lambda r: r["stat"] == "mmdagg" and r["budget"] == "b1500"), new("mmdagg", 1500), 1e-6)
