"""
synth_fig4a_all.py — all alternatives of Figure 4(a) in one run (camera-ready): VRT and MMD on D = 500, N = 5,000 per
sample, R repetitions (default 200), one CSV.

Angular alternatives, exactly the construction of the submitted figure (synthetic_vrt_sweep.py / synthetic_mmd_sweep.py,
samplers copied verbatim below, P from RandomState(rep), Q from RandomState(rep + 1000)): sample A = vMF(kappa = 250)
and sample B =
  null_vmf    vMF(250)                                   calibration
  conc_0.3    vMF(250 + 0.3 sqrt(D))                      concentration shift
  mean_2deg   vMF(250) rotated by 2 degrees in (x1, x2)  mean-direction shift
  mix_0.05    0.95 vMF(250) + 0.05 vMF(750)               mixture contamination
Radial alternatives, exactly the construction of synth_radial_fig4a.py / the rebuttal (RandomState(20260724 + rep), S5 drawn
and skipped): lognormal radii (median 10, log-sd 0.05) on vMF(250) directions, B with
  S0_null, S1_radial_mean (radii x 1.03), S2_radial_tail (log-sd 0.20), S3_angular (kappa + 0.3 sqrt(D)), S4 (S3 + S1)
Tests on the points (angular alternatives are unit vectors; radial samples are l2-normalized first):
  VRT  paper implementation (supplementary methods.py, copied in synth_radial_control.py): k = 60, M = 2,000, n_qtr = 3,000,
       500 permutations, seed = rep
  MMD  methods_v2.mmd_test via mmd_test_fast (identical p): 1,500 points per sample, median heuristic, 1,000 permutations,
       seed = rep
  NORM-MED, NORM-DISP on the radial samples (rebuttal norm tests, 500 permutations)
Test randomness (reference subsample, permutations, MMD subsample) uses seed TEST_SEED + rep, independent of the data seeds.
The submitted sweeps used seed = rep for both the data and the tests; --check_legacy repeats the first 50 repetitions with
those seeds (columns VRT_legacy, MMD_legacy and, if methods_v2 is importable from /home3/p318770/venvs/overfitting,
VRT_v2_legacy), which must give 43/4/27 (paper VRT), 11/41/49 (MMD) and 42/4/27 (methods_v2 VRT) as in the submitted figure
and in the run of October 7 (job 32570716). The radial VRT p-values of the first 3 repetitions are compared with
synth_radial_results.csv (rebuttal settings), and the S4 p-values must equal the S3 p-values.
Usage (cluster CPU, from the s50 code folder; single-threaded BLAS per worker):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python synth_fig4a_all.py --jobs 8 --reps 200 --check_legacy \
      --out <BF>/s50/synth/fig4a_all.csv
"""
import argparse, csv, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synth_radial_control as src
from mmd_tst_fast import mmd_test_fast

D, N = 500, 5000
KB, SQD = D * 0.5, np.sqrt(D)
TEST_SEED = 10 ** 6                        # test randomness independent of the data seeds (rep and 20260724 + rep)
RADIAL = ["S0_null", "S1_radial_mean", "S2_radial_tail", "S3_angular", "S4_angular_plus_radial"]


# ── samplers of synthetic_vrt_sweep.py / synthetic_mmd_sweep.py, verbatim ──────────────────────────────────────────────
def sample_vmf(n, D, kappa, rng):
    mu = np.zeros(D); mu[0] = 1.0
    d = D; b = (-2*kappa + np.sqrt(4*kappa**2 + (d-1)**2)) / (d-1)
    x0 = (1 - b) / (1 + b); c = kappa * x0 + (d-1) * np.log(1 - x0**2)
    samples = np.empty((n, D)); idx = 0
    while idx < n:
        z = rng.beta((d-1)/2, (d-1)/2, size=n)
        w = (1 - (1+b)*z) / (1 - (1-b)*z)
        u = rng.uniform(size=n)
        accept = kappa*w + (d-1)*np.log(1-x0*w) - c >= np.log(u)
        ws = w[accept]; need = min(len(ws), n-idx)
        v = rng.standard_normal((need, D-1))
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        sq = np.sqrt(np.maximum(1 - ws[:need]**2, 0))[:, None]
        samples[idx:idx+need] = np.concatenate([ws[:need,None], sq*v], axis=1)
        idx += need
    return samples


def sample_vmf_rotated(n, D, kappa, mu2, rng):
    samples = sample_vmf(n, D, kappa, rng)
    e1 = np.zeros(D); e1[0] = 1.0
    if np.allclose(mu2, e1): return samples
    if np.allclose(mu2, -e1):
        samples[:, 0] *= -1; return samples
    u_vec = mu2 - e1; u_vec = u_vec / np.linalg.norm(u_vec)
    return samples - 2 * np.outer(samples @ u_vec, u_vec)


def sample_vmf_mixture(n, D, kappa_base, eps, kappa_spike, rng):
    mu = np.zeros(D); mu[0] = 1.0
    n_spike = rng.binomial(n, eps)
    n_base = n - n_spike
    parts = []
    if n_base > 0: parts.append(sample_vmf(n_base, D, kappa_base, rng))
    if n_spike > 0: parts.append(sample_vmf(n_spike, D, kappa_spike, rng))
    out = np.concatenate(parts, 0)
    return out[rng.permutation(n)]
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

MU2 = np.zeros(D); MU2[0] = np.cos(np.radians(2.0)); MU2[1] = np.sin(np.radians(2.0))
ANGULAR = [("null_vmf", lambda rng: sample_vmf(N, D, KB, rng)),
           ("conc_0.3", lambda rng: sample_vmf(N, D, KB + 0.3 * SQD, rng)),
           ("mean_2deg", lambda rng: sample_vmf_rotated(N, D, KB, MU2, rng)),
           ("mix_0.05", lambda rng: sample_vmf_mixture(N, D, KB, 0.05, KB * 3, rng))]

V2 = None


def load_v2():
    global V2
    try:
        sys.path.insert(0, "/home3/p318770/venvs/overfitting")
        import methods_v2
        V2 = methods_v2
    except Exception as e:                                   # noqa
        V2 = False
        print(f"methods_v2 not importable ({e}); the v2 check is skipped", flush=True)


def run_rep(args):
    rep, check_legacy, ref = args
    t0 = time.time()
    rows, checks = [], []
    ts = TEST_SEED + rep
    # angular alternatives (sweep construction)
    P = sample_vmf(N, D, KB, np.random.RandomState(rep))
    for name, qfn in ANGULAR:
        Q = qfn(np.random.RandomState(rep + 1000))
        _, pv = src.vrt_test(P, Q, D=D, k=60, M=2000, n_qtr=min(8000, N - 2000), n_perm=500, seed=ts)
        _, pm = mmd_test_fast(P, Q, n_sub=1500, n_perm=1000, seed=ts, backend="numpy")
        rows += [(name, "VRT", rep, pv), (name, "MMD", rep, pm)]
        if check_legacy and rep < 50:                   # the seeds of the submitted sweeps (seed = rep for the tests)
            _, pvl = src.vrt_test(P, Q, D=D, k=60, M=2000, n_qtr=min(8000, N - 2000), n_perm=500, seed=rep)
            _, pml = mmd_test_fast(P, Q, n_sub=1500, n_perm=1000, seed=rep, backend="numpy")
            rows += [(name, "VRT_legacy", rep, pvl), (name, "MMD_legacy", rep, pml)]
            if V2:
                _, pv2 = V2.vrt_test(P, Q, D=D, k=60, M=2000, n_qtr=min(8000, N - 2000), n_perm=500, seed=rep)
                rows.append((name, "VRT_v2_legacy", rep, pv2))
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
        B_norm = src.normalize(B)
        _, pv = src.vrt_test(A_norm, B_norm, D=D, k=60, M=2000, n_qtr=N - 2000, n_perm=500, seed=ts)
        _, pm = mmd_test_fast(A_norm, B_norm, n_sub=1500, n_perm=1000, seed=ts, backend="numpy")
        normsB = np.linalg.norm(B, axis=1)
        p_nm = src.median_perm_test(normsA, normsB, 500, np.random.RandomState(ts))
        m = np.median(np.log(np.concatenate([normsA, normsB])))
        p_nd = src.median_perm_test(np.abs(np.log(normsA) - m), np.abs(np.log(normsB) - m), 500, np.random.RandomState(ts + 1))
        rows += [(name, "VRT", rep, pv), (name, "MMD", rep, pm), (name, "NORM-MED", rep, p_nm), (name, "NORM-DISP", rep, p_nd)]
        if ref is not None and rep < 3:
            M_ = min(2000, N // 3)
            _, p_ref = src.vrt_test(A_norm, B_norm, D=D, k=60, M=M_, n_qtr=N - M_, n_perm=500, seed=rep)
            checks.append((name, rep, p_ref, ref.get((name, "VRT", rep))))
    return rows, checks, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--quick", action="store_true", help="2 repetitions")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="fig4a_all.csv")
    ap.add_argument("--ref", default="synth_radial_results.csv", help="rebuttal radial results for the sample check")
    ap.add_argument("--check_legacy", action="store_true", help="first 50 repetitions also with the seeds of the submitted sweeps")
    args = ap.parse_args()
    n_rep = 2 if args.quick else args.reps
    if args.check_legacy:
        load_v2()
    ref = None
    if os.path.exists(args.ref):
        ref = {(r["cond"], r["test"], int(r["rep"])): float(r["p"]) for r in csv.DictReader(open(args.ref))}
    tasks = [(rep, args.check_legacy, ref) for rep in range(n_rep)]
    print(f"# synth_fig4a_all | D={D} N={N} reps={n_rep} | VRT k=60 M=2000 n_qtr=3000 n_perm=500 | MMD n_sub=1500 n_perm=1000 | test seed {TEST_SEED} + rep", flush=True)
    all_rows, all_checks = [], []
    if args.jobs > 1:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            for i, (rows, checks, dt) in enumerate(pool.imap_unordered(run_rep, tasks)):
                all_rows += rows; all_checks += checks
                if (i + 1) % 10 == 0 or i + 1 == n_rep:
                    print(f"  {i + 1}/{n_rep} repetitions done (last {dt:.0f} s)", flush=True)
    else:
        for i, t in enumerate(tasks):
            rows, checks, dt = run_rep(t)
            all_rows += rows; all_checks += checks
            print(f"  rep {t[0]} done ({i + 1}/{n_rep}, {dt:.0f} s)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cond", "test", "rep", "p"])
        w.writerows(sorted(all_rows, key=lambda r: (r[0], r[1], r[2])))
    bad = [r for r in all_rows if not (isinstance(r[3], float) and r[3] == r[3] and 0.0 <= r[3] <= 1.0)]
    print(f"\np-values that are not numbers in [0, 1]: {len(bad)}" + (f"  FIRST {bad[:5]}" if bad else "  (none)"))
    if all_checks:
        nb = [c for c in all_checks if c[3] is None or abs(c[2] - c[3]) > 1e-12]
        print(f"radial sample check against {args.ref}: {len(all_checks) - len(nb)}/{len(all_checks)} VRT p-values reproduced"
              + ("" if not nb else f"; MISMATCH {nb[:3]}"))
    tests = ["VRT", "MMD", "NORM-MED", "NORM-DISP", "VRT_legacy", "MMD_legacy", "VRT_v2_legacy"]
    conds = [c for c, _ in ANGULAR] + RADIAL

    def table(title, reps):
        print(f"\n{title}")
        print(f"{'cond':24s} " + " ".join(f"{t:>10s}" for t in tests))
        for c in conds:
            cells = []
            for t in tests:
                ps = [r[3] for r in all_rows if r[0] == c and r[1] == t and r[2] in reps]
                cells.append(f"{sum(p < 0.05 for p in ps):>4d}/{len(ps):<3d}" if ps else "")
            print(f"{c:24s} " + " ".join(f"{x:>10s}" for x in cells))
    table(f"rejections at alpha 0.05, all {n_rep} repetitions", set(range(n_rep)))
    table("rejections at alpha 0.05, first 50 repetitions (legacy seeds must give VRT_legacy 43/4/27, MMD_legacy 11/41/49, VRT_v2_legacy 42/4/27)", set(range(50)))
    s3 = {r[2]: r[3] for r in all_rows if r[0] == "S3_angular" and r[1] == "VRT"}
    s4 = {r[2]: r[3] for r in all_rows if r[0] == "S4_angular_plus_radial" and r[1] == "VRT"}
    if s3:
        print(f"\nS4 vs S3, VRT: max |dp| = {max(abs(s3[k] - s4[k]) for k in s3):.3g} over {len(s3)} repetitions")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
