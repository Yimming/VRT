"""
synth_fig4a_clusters.py — class-cluster alternatives for Figure 4(a) at matched budget (camera-ready, October 8).

Embedding model: C cluster centres drawn uniformly on S^499 (D = 500), fixed per repetition and shared by both samples;
every point draws a cluster uniformly and a direction from vMF(kappa_w) around its centre. Sample A plays the training
set, sample B the test set, N = 5,000 points each. The alternatives change one sample only:
  null              both samples kappa_w = kappa0 (default 2,500, about 26 degrees from the centre)
  tight_<rel>       A has kappa_w = kappa0 (1 + rel), B has kappa0: the training points sit tighter in every class
  memo_<rho>        a fraction rho of A's points has kappa_w = 3 kappa0, the rest kappa0; B all kappa0
  shift_<deg>       B's centres are rotated by <deg> degrees in random directions (per-class location shift)
Tests at matched budget, as in synth_fig4a_matched.py: VRT with k = 60, M = 2,000 reference points from A, the remaining
3,000 points of A and all 5,000 points of B as queries; MMD with a Gaussian kernel and the median heuristic on all 5,000
points of each sample; 2,000 permutations for both; test randomness seeded by TEST_SEED + rep. The single-cluster
alternatives (global concentration, mean direction, mixture, radial) of the same figure come from fig4a_matched.csv.
Output: CSV with cond,C,test,rep,p and a rejection table per C on stdout.
Usage:  python synth_fig4a_clusters.py --jobs 6 --reps 200 --out fig_data_s50/fig4a_clusters.csv
        python synth_fig4a_clusters.py --quick            (2 repetitions)
Conditions are set in CONDS below (C=10 figure set, the tightening sweep for the appendix, and the C=100 replication).
"""
import argparse, csv, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synth_radial_control as src
from mmd_tst_fast import mmd_test_fast
from synth_fig4a_all import sample_vmf_rotated, D, N, TEST_SEED

KAPPA0 = 2500.0
CONDS = [                                   # (C, name, kind, param)
    (10, "null", "null", 0.0),
    (10, "tight_0.05", "tight", 0.05),
    (10, "memo_0.10", "memo", 0.10),
    (10, "shift_4", "shift", 4.0),
    (10, "tight_0.03", "tight", 0.03),      # tightening sweep for the appendix
    (10, "tight_0.07", "tight", 0.07),
    (10, "tight_0.10", "tight", 0.10),
    (10, "tight_0.30", "tight", 0.30),
    (100, "null", "null", 0.0),             # 100 classes, replication
    (100, "tight_0.05", "tight", 0.05),
    (100, "memo_0.10", "memo", 0.10),
    (100, "shift_4", "shift", 4.0),
]


def uniform_dirs(n, rng):
    g = rng.standard_normal((n, D))
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def rotate_small(centres, delta_deg, rng):
    g = rng.standard_normal(centres.shape)
    g -= np.sum(g * centres, axis=1, keepdims=True) * centres
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    a = np.radians(delta_deg)
    return np.cos(a) * centres + np.sin(a) * g


def cluster_sample(n, centres, kappas, rng):
    """n points: cluster drawn uniformly, direction vMF(kappa of the point) around the cluster centre"""
    C = len(centres)
    lab = rng.randint(0, C, n)
    out = np.empty((n, D))
    for c in range(C):
        idx = np.where(lab == c)[0]
        if len(idx) == 0:
            continue
        for kap in np.unique(kappas[idx]):
            sub = idx[kappas[idx] == kap]
            out[sub] = sample_vmf_rotated(len(sub), D, float(kap), centres[c], rng)
    return out


def make(C, kind, param, rep):
    """samples A (training) and B (test); the centres and A's draws depend on (rep, C), B's draws on rep + 1000"""
    rng = np.random.RandomState(rep * 1000 + C)
    rngB = np.random.RandomState(rep * 1000 + C + 500)
    centres = uniform_dirs(C, rng)
    kA = np.full(N, KAPPA0)
    kB = np.full(N, KAPPA0)
    cB = centres
    if kind == "tight":
        kA[:] = KAPPA0 * (1 + param)
    elif kind == "memo":
        kA[:int(round(param * N))] = 3 * KAPPA0
    elif kind == "shift":
        cB = rotate_small(centres, param, rng)
    A = cluster_sample(N, centres, kA, rng)
    A = A[rng.permutation(N)]
    B = cluster_sample(N, cB, kB, rngB)
    return A, B


def run_rep(args):
    rep, n_perm, conds = args
    t0 = time.time()
    ts = TEST_SEED + rep
    rows = []
    for C, name, kind, param in conds:
        A, B = make(C, kind, param, rep)
        _, pv = src.vrt_test(A, B, D=D, k=60, M=2000, n_qtr=N - 2000, n_perm=n_perm, seed=ts)
        _, pm = mmd_test_fast(A, B, n_sub=N, n_perm=n_perm, seed=ts, backend="numpy")
        rows += [(name, C, "VRT", rep, pv), (name, C, "MMD", rep, pm)]
    return rows, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--only", default="", help="comma list of condition names to run (default all)")
    ap.add_argument("--rep_offset", type=int, default=0, help="first repetition number (the pilot used 0 to 7; use 200 for fresh samples)")
    ap.add_argument("--out", default="fig4a_clusters.csv")
    args = ap.parse_args()
    n_rep = 2 if args.quick else args.reps
    conds = [c for c in CONDS if not args.only or c[1] in args.only.split(",")]
    print(f"# synth_fig4a_clusters | D={D} N={N} kappa0={KAPPA0:.0f} reps={n_rep} from {args.rep_offset} | {len(conds)} conditions | VRT k=60 M=2000 | "
          f"MMD all points, median heuristic | {args.n_perm} permutations | test seed {TEST_SEED} + rep", flush=True)
    tasks = [(rep, args.n_perm, conds) for rep in range(args.rep_offset, args.rep_offset + n_rep)]
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
            rows, dt = run_rep(t)
            all_rows += rows
            print(f"  rep {t[0]} done ({i + 1}/{n_rep}, {dt:.0f} s)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cond", "C", "test", "rep", "p"])
        w.writerows(sorted(all_rows, key=lambda r: (r[1], r[0], r[2], r[3])))
    bad = [r for r in all_rows if not (isinstance(r[4], float) and r[4] == r[4] and 0.0 <= r[4] <= 1.0)]
    print(f"\np-values that are not numbers in [0, 1]: {len(bad)}" + (f"  FIRST {bad[:5]}" if bad else "  (none)"))
    for C in sorted({c[0] for c in conds}):
        print(f"\nrejections at alpha 0.05, C = {C} classes, {n_rep} repetitions")
        print(f"{'cond':14s} {'VRT':>10s} {'MMD':>10s}")
        for Cc, name, _, _ in conds:
            if Cc != C:
                continue
            cells = []
            for t in ("VRT", "MMD"):
                ps = [r[4] for r in all_rows if r[0] == name and r[1] == C and r[2] == t]
                cells.append(f"{sum(p < 0.05 for p in ps):>4d}/{len(ps):<3d}")
            print(f"{name:14s} " + " ".join(f"{x:>10s}" for x in cells))
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
