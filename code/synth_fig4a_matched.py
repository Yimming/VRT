"""
synth_fig4a_matched.py — Figure 4(a) with VRT and MMD at the same budget (camera-ready, October 8).

Same samples as synth_fig4a_all.py (same data seeds, same samplers, same five alternatives and two nulls, D = 500, N = 5,000
per sample, 200 repetitions), but both tests see the same points and use the same number of permutations:
  budget n per sample   5,000 (all points), 1,500, 500          (--budgets)
  VRT                   reference M and k scale with the budget as in the paper's two tiers (k/M = 0.03):
                        n = 5,000: M = 2,000, k = 60; 1,500: M = 500, k = 15; 500: M = 166, k = 5;
                        training queries = the n - M remaining points of A, test queries = all n points of B
  MMD                   Gaussian kernel, median heuristic, all n points of each sample (no further subsample)
  permutations          2,000 for both tests (--n_perm)
  NORM-MED, NORM-DISP   on the norms of the radial samples at every budget (500 permutations, as in the rebuttal)
For a budget below 5,000 the same random subset of each sample (seeded by TEST_SEED + rep) is given to both tests and to
every alternative of a repetition. The run of October 7 (synth_fig4a_all.py: VRT on all points with 500 permutations, MMD
on a 1,500-point subsample with 1,000 permutations) is kept as fig4a_all.csv for the record; this run replaces it in the
figure.
Output: CSV with cond,test,budget,rep,p and, on stdout, rejections at alpha 0.05 per budget.
Usage (cluster CPU, from the s50 code folder, which holds synth_fig4a_all.py, synth_radial_control.py, mmd_tst_fast.py):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python synth_fig4a_matched.py --jobs 16 --reps 200 --out <BF>/s50/synth/fig4a_matched.csv
  python synth_fig4a_matched.py --quick --budgets 500,1500     (2 repetitions, smoke test)
"""
import argparse, csv, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synth_radial_control as src
from mmd_tst_fast import mmd_test_fast
from synth_fig4a_all import ANGULAR, RADIAL, D, N, TEST_SEED, sample_vmf, KB   # samplers and conditions, verbatim

VRT_TIER = {5000: (2000, 60), 1500: (500, 15), 500: (166, 5)}      # budget -> (M, k), k/M = 0.03 as in the paper's tiers


def vrt_setting(n):
    if n in VRT_TIER:
        return VRT_TIER[n]
    M = max(n // 3, 20)
    return M, max(int(round(0.03 * M)), 2)


def run_rep(args):
    rep, budgets, n_perm = args
    t0 = time.time()
    ts = TEST_SEED + rep
    rows = []
    sub = {}
    for n in budgets:                                       # one subset per budget, shared by both samples' roles below
        sub[n] = (np.random.RandomState(ts + 7).permutation(N)[:n], np.random.RandomState(ts + 8).permutation(N)[:n])

    def both(name, P, Q, norms=None):
        for n in budgets:
            iP, iQ = sub[n]
            Pn, Qn = (P, Q) if n >= N else (P[iP], Q[iQ])
            M, k = vrt_setting(n)
            _, pv = src.vrt_test(Pn, Qn, D=D, k=k, M=M, n_qtr=n - M, n_perm=n_perm, seed=ts)
            _, pm = mmd_test_fast(Pn, Qn, n_sub=n, n_perm=n_perm, seed=ts, backend="numpy")
            rows.extend([(name, "VRT", n, rep, pv), (name, "MMD", n, rep, pm)])
            if norms is not None:
                nA, nB = norms
                nA, nB = (nA, nB) if n >= N else (nA[iP], nB[iQ])
                p_nm = src.median_perm_test(nA, nB, 500, np.random.RandomState(ts))
                m = np.median(np.log(np.concatenate([nA, nB])))
                p_nd = src.median_perm_test(np.abs(np.log(nA) - m), np.abs(np.log(nB) - m), 500, np.random.RandomState(ts + 1))
                rows.extend([(name, "NORM-MED", n, rep, p_nm), (name, "NORM-DISP", n, rep, p_nd)])

    # angular alternatives (sweep construction, data seeds as in synth_fig4a_all.py)
    P = sample_vmf(N, D, KB, np.random.RandomState(rep))
    for name, qfn in ANGULAR:
        Q = qfn(np.random.RandomState(rep + 1000))
        both(name, P, Q)
    # radial alternatives (rebuttal construction)
    rng = np.random.RandomState(20260724 + rep)
    kA = D / 2.0
    kB = kA + 0.3 * np.sqrt(D)
    uA = src.sample_vmf(N, D, src._e1(D), kA, rng)
    rA = np.exp(rng.normal(src.LOGMU, src.SIG0, N))
    A = rA[:, None] * uA
    A_norm = src.normalize(A)
    normsA = np.linalg.norm(A, axis=1)
    conds = src.build_conditions(rng, N, D, kA, kB, include_coupling=True)
    for name in RADIAL:
        u, r = conds[name]
        B = r[:, None] * u
        both(name, A_norm, src.normalize(B), norms=(normsA, np.linalg.norm(B, axis=1)))
    return rows, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--quick", action="store_true", help="2 repetitions")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--budgets", default="5000,1500,500")
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--out", default="fig4a_matched.csv")
    args = ap.parse_args()
    n_rep = 2 if args.quick else args.reps
    budgets = sorted({int(b) for b in args.budgets.split(",")}, reverse=True)
    if any(b > N or b < 100 for b in budgets):
        raise SystemExit(f"budgets must lie in [100, {N}]")
    print(f"# synth_fig4a_matched | D={D} N={N} reps={n_rep} | budgets {budgets} | VRT (M, k) "
          f"{[vrt_setting(b) for b in budgets]} | MMD on all n points, median heuristic | {args.n_perm} permutations for both"
          f" | test seed {TEST_SEED} + rep", flush=True)
    tasks = [(rep, budgets, args.n_perm) for rep in range(n_rep)]
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
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cond", "test", "budget", "rep", "p"])
        w.writerows(sorted(all_rows, key=lambda r: (r[0], r[1], -r[2], r[3])))
    bad = [r for r in all_rows if not (isinstance(r[4], float) and r[4] == r[4] and 0.0 <= r[4] <= 1.0)]
    print(f"\np-values that are not numbers in [0, 1]: {len(bad)}" + (f"  FIRST {bad[:5]}" if bad else "  (none)"))
    tests = ["VRT", "MMD", "NORM-MED", "NORM-DISP"]
    conds = [c for c, _ in ANGULAR] + RADIAL
    for n in budgets:
        print(f"\nrejections at alpha 0.05, budget {n} per sample, {n_rep} repetitions")
        print(f"{'cond':24s} " + " ".join(f"{t:>10s}" for t in tests))
        for c in conds:
            cells = []
            for t in tests:
                ps = [r[4] for r in all_rows if r[0] == c and r[1] == t and r[2] == n]
                cells.append(f"{sum(p < 0.05 for p in ps):>4d}/{len(ps):<3d}" if ps else "")
            print(f"{c:24s} " + " ".join(f"{x:>10s}" for x in cells))
        s3 = {r[3]: r[4] for r in all_rows if r[0] == "S3_angular" and r[1] == "VRT" and r[2] == n}
        s4 = {r[3]: r[4] for r in all_rows if r[0] == "S4_angular_plus_radial" and r[1] == "VRT" and r[2] == n}
        if s3:
            print(f"S4 vs S3, VRT, budget {n}: max |dp| = {max(abs(s3[k] - s4[k]) for k in s3):.3g} over {len(s3)} repetitions")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
