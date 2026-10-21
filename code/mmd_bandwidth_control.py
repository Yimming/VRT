"""
mmd_bandwidth_control.py — fixed-bandwidth control for the class-cluster alternative of Figure 4(a) (October 8).

Same samples as synth_fig4a_clusters.py (same make(), same repetition numbers), C = 10, conditions null and tight_0.05
by default. For each repetition the median-heuristic bandwidth sigma_med is computed as in mmd_test_fast (same rng draw),
and MMD is run at the bandwidths s * sigma_med for every scale s in --scales (default 0.25, 0.5, 1, 2), all on the same
5,000 + 5,000 points with the same 2,000 permutations (same permutation draws for every scale). VRT is run once per
repetition for reference. The script also records, per repetition, sigma_med and the median Euclidean distance between
points of the same class and of different classes in sample A (classes assigned by the nearest centre), so that the
bandwidth can be placed on the within-class and between-class scales.
Output: CSV with cond,scale,test,rep,p,sigma,d_within,d_between and a rejection table per condition and scale.
Usage:  python mmd_bandwidth_control.py --jobs 6 --reps 200 --rep_offset 200 --out fig_data_s50/mmd_bandwidth.csv
"""
import argparse, csv, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synth_radial_control as src
from mmd_tst_fast import median_sigma, rbf_matrix, mmd2_from_matrix
from synth_fig4a_clusters import make, uniform_dirs, KAPPA0, CONDS
from synth_fig4a_all import D, N, TEST_SEED


def mmd_scales(A, B, scales, n_perm, seed):
    """p-values of the permutation MMD at bandwidths s * sigma_med, same subsample order, bandwidth draw and permutations
    as mmd_test_fast(A, B, n_sub=N, n_perm, seed) for s = 1"""
    rng = np.random.RandomState(seed)
    X = A[rng.permutation(len(A))]
    Y = B[rng.permutation(len(B))]
    pooled = np.concatenate([X, Y])
    sigma = median_sigma(pooled, rng)
    n, m = len(X), len(Y); Nn = n + m
    perms = np.stack([rng.permutation(Nn) for _ in range(n_perm)])
    P = np.zeros((Nn, n_perm + 1), dtype=np.float64)
    P[:n, 0] = 1.0
    P[perms[:, :n].T, np.arange(1, n_perm + 1)[None, :]] = 1.0
    out = {}
    for s in scales:
        K = rbf_matrix(pooled, sigma * s, "numpy")
        val = mmd2_from_matrix(K, P, n, m)
        out[s] = float((np.sum(val[1:] >= val[0]) + 1) / (n_perm + 1))
    return out, sigma


def class_distances(A, rep, C, n_sub=1500):
    """median Euclidean distance within and between classes in sample A (classes = nearest centre)"""
    rng = np.random.RandomState(rep * 1000 + C)          # the centres of make(): same draw as in synth_fig4a_clusters.make
    centres = uniform_dirs(C, rng)
    sub = A[np.random.RandomState(rep).permutation(len(A))[:n_sub]]
    lab = np.argmax(sub @ centres.T, axis=1)
    sq = np.sum(sub ** 2, 1, keepdims=True)
    d = np.sqrt(np.maximum(sq + sq.T - 2 * sub @ sub.T, 0.0))
    same = lab[:, None] == lab[None, :]
    iu = np.triu_indices(len(sub), 1)
    return float(np.median(d[iu][same[iu]])), float(np.median(d[iu][~same[iu]]))


def run_rep(args):
    rep, scales, n_perm, conds = args
    t0 = time.time(); ts = TEST_SEED + rep; rows = []
    for C, name, kind, param in conds:
        A, B = make(C, kind, param, rep)
        dw, db = class_distances(A, rep, C)
        pm, sigma = mmd_scales(A, B, scales, n_perm, ts)
        _, pv = src.vrt_test(A, B, D=D, k=60, M=2000, n_qtr=N - 2000, n_perm=n_perm, seed=ts)
        rows.append((name, 0.0, "VRT", rep, pv, sigma, dw, db))
        for s in scales:
            rows.append((name, s, "MMD", rep, pm[s], sigma, dw, db))
    return rows, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--rep_offset", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--scales", default="0.25,0.5,1,2")
    ap.add_argument("--conds", default="null,tight_0.05", help="condition names of synth_fig4a_clusters.CONDS at C = 10")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="mmd_bandwidth.csv")
    args = ap.parse_args()
    scales = [float(s) for s in args.scales.split(",")]
    conds = [c for c in CONDS if c[0] == 10 and c[1] in args.conds.split(",")]
    n_rep = 2 if args.quick else args.reps
    print(f"# mmd_bandwidth_control | C=10 kappa0={KAPPA0:.0f} | conditions {[c[1] for c in conds]} | scales {scales} x median sigma | "
          f"{args.n_perm} permutations | reps {args.rep_offset} to {args.rep_offset + n_rep - 1}", flush=True)
    tasks = [(rep, scales, args.n_perm, conds) for rep in range(args.rep_offset, args.rep_offset + n_rep)]
    all_rows = []
    if args.jobs > 1:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            for i, (rows, dt) in enumerate(pool.imap_unordered(run_rep, tasks)):
                all_rows += rows
                if (i + 1) % 10 == 0 or i + 1 == n_rep:
                    print(f"  {i + 1}/{n_rep} repetitions done (last {dt:.0f} s)", flush=True)
    else:
        for i, t in enumerate(tasks):
            rows, dt = run_rep(t); all_rows += rows
            print(f"  rep {t[0]} done ({i + 1}/{n_rep}, {dt:.0f} s)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cond", "scale", "test", "rep", "p", "sigma", "d_within", "d_between"])
        w.writerows(sorted(all_rows, key=lambda r: (r[0], r[2], r[1], r[3])))
    sig = np.array([r[5] for r in all_rows]); dw = np.array([r[6] for r in all_rows]); db = np.array([r[7] for r in all_rows])
    print(f"\nmedian-heuristic sigma {np.median(sig):.3f} (range {sig.min():.3f} to {sig.max():.3f}); median within-class distance "
          f"{np.median(dw):.3f}, between-class {np.median(db):.3f}  (unit vectors, so the between-class value is near sqrt 2)")
    print(f"\n{'cond':12s} {'VRT':>9s} " + " ".join(f"MMD x{s:<5g}" for s in scales))
    for C, name, _, _ in conds:
        pv = [r[4] for r in all_rows if r[0] == name and r[2] == "VRT"]
        cells = [f"{sum(p < 0.05 for p in pv):>4d}/{len(pv):<3d}"]
        for s in scales:
            pm = [r[4] for r in all_rows if r[0] == name and r[2] == "MMD" and r[1] == s]
            cells.append(f"{sum(p < 0.05 for p in pm):>4d}/{len(pm):<3d}")
        print(f"{name:12s} " + " ".join(f"{c:>9s}" for c in cells))
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
