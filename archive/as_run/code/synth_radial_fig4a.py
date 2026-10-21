"""
synth_radial_fig4a.py — the two radial alternatives of Figure 4(a) (camera-ready): VRT and MMD on samples that differ only
in the scale or the spread of the norms, plus the null and the concentration alternative as checks.

Samples: exactly those of the rebuttal run synth_radial_control.py (same RandomState(20260724 + rep) stream, conditions
built with include_coupling=True as in that run, S5 drawn and skipped), D = 500, N = 5,000 per sample, 50 repetitions:
  S0 null            A and B from one law (vMF kappa = 250 directions, lognormal radii, median 10, log-sd 0.05)
  S1 radial scale    radii of B x 1.03, directions from the same law
  S2 radial spread   log-sd of the radii of B 0.05 -> 0.20 (median unchanged)
  S3 concentration   kappa_B = 250 + 0.3 sqrt(D), the concentration alternative of the paper
  S4 S3 + S1         same direction draws as S3, radii x 1.03 (after normalization identical to S3)
Tests, all on l2-normalized points, with the settings of the Figure 4(a) sweeps (synthetic_vrt_sweep.py and
synthetic_mmd_sweep.py): VRT k = 60, M = 2,000, n_qtr = 3,000, 500 permutations, seed = rep; MMD = methods_v2.mmd_test
(1,500 points per sample, median heuristic, 1,000 permutations, seed = rep), evaluated with mmd_tst_fast (identical p).
The norm tests of the rebuttal (NORM-MED, NORM-DISP on ||x||, 500 permutations) are repeated for the appendix text.
Check: for the first three repetitions VRT is also run with the rebuttal settings (M = 1,666) and compared with
synth_radial_results.csv, which confirms that the samples are the same.

Usage (cluster CPU job, from the s50 code folder, which holds synth_radial_control.py and mmd_tst_fast.py):
  python synth_radial_fig4a.py --jobs 8 --out <BF>/s50/synth/fig4a_radial.csv [--ref synth_radial_results.csv]
  python synth_radial_fig4a.py --quick    (2 repetitions, smoke test)
Output: CSV with cond,test,rep,p and a summary of rejections at alpha = 0.05 per condition and test.
"""
import argparse, csv, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_radial_control as src                      # sampler and VRT copied verbatim from the paper code
from mmd_tst_fast import mmd_test_fast

CONDS = ["S0_null", "S1_radial_mean", "S2_radial_tail", "S3_angular", "S4_angular_plus_radial"]
D, N, N_REP = 500, 5000, 50


def run_rep(args):
    rep, ref_rows, do_check = args
    t0 = time.time()
    rng = np.random.RandomState(20260724 + rep)
    kA = D / 2.0
    kB = kA + 0.3 * np.sqrt(D)
    uA = src.sample_vmf(N, D, src._e1(D), kA, rng)
    rA = np.exp(rng.normal(src.LOGMU, src.SIG0, N))
    A = rA[:, None] * uA
    A_norm = src.normalize(A)
    normsA = np.linalg.norm(A, axis=1)
    conds = src.build_conditions(rng, N, D, kA, kB, include_coupling=True)   # same stream as the rebuttal run
    rows, checks = [], []
    for name in CONDS:
        u, r = conds[name]
        B = r[:, None] * u
        B_norm = src.normalize(B)
        _, p_vrt = src.vrt_test(A_norm, B_norm, D=D, k=60, M=2000, n_qtr=N - 2000, n_perm=500, seed=rep)
        _, p_mmd = mmd_test_fast(A_norm, B_norm, n_sub=1500, n_perm=1000, seed=rep, backend="numpy")
        normsB = np.linalg.norm(B, axis=1)
        p_nm = src.median_perm_test(normsA, normsB, 500, np.random.RandomState(rep))
        m = np.median(np.log(np.concatenate([normsA, normsB])))
        p_nd = src.median_perm_test(np.abs(np.log(normsA) - m), np.abs(np.log(normsB) - m), 500,
                                    np.random.RandomState(rep + 1))
        rows += [(name, "VRT", rep, p_vrt), (name, "MMD", rep, p_mmd), (name, "NORM-MED", rep, p_nm), (name, "NORM-DISP", rep, p_nd)]
        if do_check and ref_rows is not None:
            M_ = min(2000, N // 3)
            _, p_ref = src.vrt_test(A_norm, B_norm, D=D, k=60, M=M_, n_qtr=N - M_, n_perm=500, seed=rep)
            checks.append((name, rep, p_ref, ref_rows.get((name, "VRT", rep))))
    return rows, checks, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rep", type=int, default=N_REP)
    ap.add_argument("--quick", action="store_true", help="2 repetitions")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="fig4a_radial.csv")
    ap.add_argument("--ref", default="synth_radial_results.csv", help="rebuttal results for the sample check")
    args = ap.parse_args()
    n_rep = 2 if args.quick else args.n_rep
    ref = None
    if os.path.exists(args.ref):
        ref = {(r["cond"], r["test"], int(r["rep"])): float(r["p"]) for r in csv.DictReader(open(args.ref))}
    tasks = [(rep, ref, rep < 3) for rep in range(n_rep)]
    print(f"# synth_radial_fig4a | D={D} N={N} reps={n_rep} | VRT k=60 M=2000 n_qtr=3000 n_perm=500 | MMD n_sub=1500 n_perm=1000",
          flush=True)
    all_rows, all_checks = [], []
    if args.jobs > 1:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            for i, (rows, checks, dt) in enumerate(pool.imap_unordered(run_rep, tasks)):
                all_rows += rows; all_checks += checks
                print(f"  rep done ({i + 1}/{n_rep}, {dt:.0f} s)", flush=True)
    else:
        for i, t in enumerate(tasks):
            rows, checks, dt = run_rep(t)
            all_rows += rows; all_checks += checks
            print(f"  rep {t[0]} done ({i + 1}/{n_rep}, {dt:.0f} s)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cond", "test", "rep", "p"]); w.writerows(sorted(all_rows, key=lambda r: (r[0], r[1], r[2])))
    if all_checks:
        bad = [c for c in all_checks if c[3] is None or abs(c[2] - c[3]) > 1e-12]
        print(f"\nsample check against {args.ref}: {len(all_checks) - len(bad)}/{len(all_checks)} VRT p-values reproduced"
              + ("" if not bad else f"; MISMATCH {bad[:3]}"))
    print(f"\n{'cond':24s} " + " ".join(f"{t:>10s}" for t in ("VRT", "MMD", "NORM-MED", "NORM-DISP")) + "   (rejections of {n_rep} at alpha 0.05)".format(n_rep=n_rep))
    for c in CONDS:
        cells = []
        for t in ("VRT", "MMD", "NORM-MED", "NORM-DISP"):
            ps = [r[3] for r in all_rows if r[0] == c and r[1] == t]
            cells.append(f"{sum(p < 0.05 for p in ps):>4d}/{len(ps):<3d}")
        print(f"{c:24s} " + " ".join(f"{x:>10s}" for x in cells))
    s3 = {r[2]: r[3] for r in all_rows if r[0] == "S3_angular" and r[1] == "VRT"}
    s4 = {r[2]: r[3] for r in all_rows if r[0] == "S4_angular_plus_radial" and r[1] == "VRT"}
    print(f"S4 vs S3, VRT: max |dp| = {max(abs(s3[k] - s4[k]) for k in s3):.3g} over {len(s3)} repetitions")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
