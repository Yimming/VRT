#!/usr/bin/env python3
"""
synth_radial_control.py — radial / tail-only synthetic controls for VRT scope
=============================================================================
Answers nxR4 Q3 ("cases where train and test differ primarily through radial
or tail behavior while preserving angular geometry") with a direct experiment.

Fidelity guarantees:
  * VRT code (log_cap_volume / compute_chat / vrt_test) copied VERBATIM from
    the supplementary src/methods.py.
  * vMF sampler copied VERBATIM from scripts/synthetic_comparison.py.
  * Same settings as the paper's synthetic experiments (App D / C):
    D=500, N=5,000 per sample, kappa_A = D/2 = 250, dkappa = 0.3*sqrt(D),
    VRT with k=60, M=min(2000, N//3), n_perm=500, alpha=0.05, 50 reps.

Conditions — each modifies exactly ONE property of sample B (sample A fixed):
  S0 null            identical laws (calibration)
  S1 radial-mean     all radii of B scaled x1.03; angular law identical
  S2 radial-tail     radial log-sd 0.05 -> 0.20 in B; the lognormal median is
                     independent of sigma, so the radial MEDIAN is matched
                     exactly (pure tail/dispersion change)
  S3 angular         vMF concentration shift dkappa = 0.3*sqrt(D) — the
                     paper's positive control (App D reports ~84% for VRT)
  S4 angular+radial  S3's angular shift AND S1's radial scaling, built from
                     the SAME direction draws as S3. After l2-normalization
                     the inputs coincide up to float roundoff, so VRT's
                     p-values must match S3's (max |dp| printed).
  S5 coupling        [--include-coupling] radii of B rank-coupled to the
                     angle from the mean direction, with BOTH marginals
                     matched exactly. All marginal channels (angular law,
                     radial law) are unchanged; only the joint law differs.
                     Blind spot for VRT and for the norm channel alike —
                     include only to preempt the "magnitude-dependent"
                     variant of the question with an explicit scope
                     statement.

Tests per condition (all permutation-based, shared seed within a repetition):
  VRT        paper protocol on l2-normalized points
  NORM-MED   median-difference permutation test on raw norms ||x||
             (the ladder's rung-1 protocol)
  NORM-DISP  median-difference permutation test on |log||x|| - m|,
             m = median of pooled log-norms (dispersion/tail read-out of
             the same channel)

Expected pattern (cite in the rebuttal after confirming):
  cond               VRT        NORM-MED   NORM-DISP
  S0 null            ~5%        ~5%        ~5%
  S1 radial-mean     ~5%        ~100%      ~5% (symmetric split)
  S2 radial-tail     ~5%        ~5%        ~100%
  S3 angular         high(~84%) ~5%        ~5%
  S4 ang.+radial     == S3      ~100%      —      (max|dp| ~ 0)
  S5 coupling        ~5%        ~5%        ~5%    (out of scope for ALL
                                                   marginal tests; scope stmt)

Usage:
  python3 synth_radial_control.py --quick            # 10 reps smoke (~10 min)
  python3 synth_radial_control.py --jobs 8           # full 50 reps
  python3 synth_radial_control.py --include-coupling # add S5

Cluster etiquette (do NOT run the full version on the login node):
  cat > job_synth_radial.sh <<'EOF'
  #!/bin/bash
  #SBATCH --job-name=synthrad
  #SBATCH --partition=regular
  #SBATCH --cpus-per-task=8
  #SBATCH --time=04:00:00
  #SBATCH --mem=16G
  #SBATCH --output=synthrad_%j.out
  python3 synth_radial_control.py --jobs 8 --include-coupling
  EOF
  sbatch job_synth_radial.sh

Output: markdown summary on stdout + synth_radial_results.csv
        (columns: cond,test,rep,p — one row per condition x test x rep).
"""

import argparse
import csv
import sys
import time

import numpy as np
from scipy.special import betainc, betaln

ALPHA = 0.05
LOGMU = np.log(10.0)   # baseline radial law: lognormal(median 10, log-sd 0.05)
SIG0 = 0.05
SIG_TAIL = 0.20        # S2: log-sd x4, median unchanged
RAD_SCALE = 1.03       # S1/S4: +3% radial location shift

# ══════════════════════════════════════════════════════════════════════
# VRT — copied VERBATIM from supplementary src/methods.py (do not edit)
# ══════════════════════════════════════════════════════════════════════

def log_cap_volume(theta, D):
    """Log spherical cap volume, numerically stable for any D."""
    sin2 = np.sin(theta)**2
    I = betainc((D-1)/2.0, 0.5, sin2)
    V = np.where(theta <= np.pi/2, 0.5*I, 1.0 - 0.5*I)
    safe = V > 1e-300
    log_V = np.empty_like(V)
    log_V[safe] = np.log(V[safe])
    if np.any(~safe):
        a = (D-1) / 2.0
        const = -np.log(2.0) - np.log(a) - betaln(a, 0.5)
        sin_vals = np.maximum(np.sin(theta[~safe]), 1e-300)
        log_V[~safe] = const + (D-1) * np.log(sin_vals)
    return log_V


def compute_chat(query, ref, D, k, batch_size=1024):
    """Compute per-point VRT scores for query points against reference."""
    if k > ref.shape[0]:
        raise ValueError(f"k={k} exceeds reference size M={ref.shape[0]}")
    Nq = query.shape[0]
    theta_all = []
    for s in range(0, Nq, batch_size):
        e = min(s + batch_size, Nq)
        cos = query[s:e] @ ref.T
        pidx = np.argpartition(-cos, k - 1, axis=1)[:, :k]
        top = np.take_along_axis(cos, pidx, axis=1)
        top = np.take_along_axis(top, np.argsort(-top, axis=1), axis=1)
        theta_all.append(np.arccos(np.clip(top, -1, 1)))
    theta = np.concatenate(theta_all, 0)
    L = log_cap_volume(theta, D)
    dL = np.diff(L, axis=1)
    r = np.arange(1, k, dtype=np.float64)
    Z = np.maximum(r[None, :] * dL, 0.0)
    return np.maximum(Z.mean(axis=1), 1e-12)


def vrt_test(emb_tr, emb_te, D, k=60, M=2000, n_qtr=8000,
             n_perm=2000, seed=42):
    """VRT two-sample test with training embeddings as reference."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(emb_tr))
    ref = emb_tr[perm[:M]]
    q_tr = emb_tr[perm[M:M + min(n_qtr, len(emb_tr) - M)]]
    c_tr = compute_chat(q_tr, ref, D, k)
    c_te = compute_chat(emb_te, ref, D, k)
    obs = np.median(c_te) - np.median(c_tr)
    pooled = np.concatenate([c_tr, c_te])
    na = len(c_tr)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled))
        if abs(np.median(pooled[p[na:]]) - np.median(pooled[p[:na]])) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)


# ══════════════════════════════════════════════════════════════════════
# vMF sampler — copied VERBATIM from scripts/synthetic_comparison.py
# ══════════════════════════════════════════════════════════════════════

def sample_uniform(n, D, rng):
    x = rng.standard_normal((n, D))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def sample_vmf(n, D, mu, kappa, rng):
    if kappa < 1e-10:
        return sample_uniform(n, D, rng)
    d = D
    b = (-2*kappa + np.sqrt(4*kappa**2 + (d-1)**2)) / (d-1)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d-1) * np.log(1 - x0**2)
    samples = np.empty((n, D))
    idx = 0
    batch = max(n, 256)
    while idx < n:
        z = rng.beta((d-1)/2, (d-1)/2, size=batch)
        w = (1 - (1+b)*z) / (1 - (1-b)*z)
        u = rng.uniform(size=batch)
        accept = kappa*w + (d-1)*np.log(1-x0*w) - c >= np.log(u)
        ws = w[accept]
        needed = min(len(ws), n - idx)
        v = rng.standard_normal((needed, D-1))
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        samples[idx:idx+needed, 0] = ws[:needed]
        samples[idx:idx+needed, 1:] = v * np.sqrt(1 - ws[:needed]**2)[:, None]
        idx += needed
    e1 = np.zeros(D); e1[0] = 1.0
    if np.allclose(mu, e1):
        return samples
    u_vec = e1 + mu; u_vec = u_vec / np.linalg.norm(u_vec)
    return samples - 2 * np.outer(samples @ u_vec, u_vec) + 2 * np.outer(samples @ e1, mu)


# ══════════════════════════════════════════════════════════════════════
# Norm-channel tests (ladder rung-1 protocol + dispersion variant)
# ══════════════════════════════════════════════════════════════════════

def median_perm_test(a, b, n_perm, rng):
    """Two-sided permutation test on the difference of medians (rung-1)."""
    obs = np.median(b) - np.median(a)
    pooled = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled))
        if abs(np.median(pooled[p[na:]]) - np.median(pooled[p[:na]])) >= abs(obs):
            count += 1
    return (count + 1) / (n_perm + 1)


def normalize(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


# ══════════════════════════════════════════════════════════════════════
# Conditions
# ══════════════════════════════════════════════════════════════════════

def build_conditions(rng, N, D, kA, kB, include_coupling):
    """Return dict: name -> (directions, radii) for sample B."""
    conds = {}

    u = sample_vmf(N, D, _e1(D), kA, rng)
    r = np.exp(rng.normal(LOGMU, SIG0, N))
    conds['S0_null'] = (u, r)

    u = sample_vmf(N, D, _e1(D), kA, rng)
    r = np.exp(rng.normal(LOGMU, SIG0, N)) * RAD_SCALE
    conds['S1_radial_mean'] = (u, r)

    u = sample_vmf(N, D, _e1(D), kA, rng)
    r = np.exp(rng.normal(LOGMU, SIG_TAIL, N))
    conds['S2_radial_tail'] = (u, r)

    # S3 / S4 share direction draws and base radii -> invariance check
    u34 = sample_vmf(N, D, _e1(D), kB, rng)
    r34 = np.exp(rng.normal(LOGMU, SIG0, N))
    conds['S3_angular'] = (u34, r34)
    conds['S4_angular_plus_radial'] = (u34, r34 * RAD_SCALE)

    if include_coupling:
        u5 = sample_vmf(N, D, _e1(D), kA, rng)
        r5 = np.exp(rng.normal(LOGMU, SIG0, N))
        r5c = np.empty(N)
        # larger radius <-> closer to mean direction; both marginals exact
        r5c[np.argsort(u5[:, 0])] = np.sort(r5)
        conds['S5_coupling'] = (u5, r5c)

    return conds


def _e1(D):
    e1 = np.zeros(D)
    e1[0] = 1.0
    return e1


def run_rep(task):
    rep, N, D, kA, dk, n_perm, include_coupling = task
    t0 = time.time()
    rng = np.random.RandomState(20260724 + rep)
    kB = kA + dk

    # sample A (fixed across conditions within the repetition)
    uA = sample_vmf(N, D, _e1(D), kA, rng)
    rA = np.exp(rng.normal(LOGMU, SIG0, N))
    A = rA[:, None] * uA
    A_norm = normalize(A)
    normsA = np.linalg.norm(A, axis=1)

    conds = build_conditions(rng, N, D, kA, kB, include_coupling)
    M_ = min(2000, N // 3)               # as in synthetic_comparison.py
    seed = rep                            # as in synthetic_comparison.py;
                                          # shared across conditions so that
                                          # S3/S4 use identical ref/perm draws
    rows = []
    for name, (u, r) in conds.items():
        B = r[:, None] * u
        _, p_vrt = vrt_test(A_norm, normalize(B), D=D, k=60, M=M_,
                            n_qtr=N - M_, n_perm=n_perm, seed=seed)
        normsB = np.linalg.norm(B, axis=1)
        p_nm = median_perm_test(normsA, normsB, n_perm,
                                np.random.RandomState(seed))
        m = np.median(np.log(np.concatenate([normsA, normsB])))
        p_nd = median_perm_test(np.abs(np.log(normsA) - m),
                                np.abs(np.log(normsB) - m), n_perm,
                                np.random.RandomState(seed + 1))
        rows += [(name, 'VRT', rep, p_vrt),
                 (name, 'NORM-MED', rep, p_nm),
                 (name, 'NORM-DISP', rep, p_nd)]
    return rows, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-rep', type=int, default=50)
    ap.add_argument('--quick', action='store_true', help='10 reps smoke run')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--D', type=int, default=500)
    ap.add_argument('--N', type=int, default=5000)
    ap.add_argument('--n-perm', type=int, default=500,
                    help='as in the paper synthetic scripts (default 500)')
    ap.add_argument('--include-coupling', action='store_true')
    ap.add_argument('--out', default='synth_radial_results.csv')
    args = ap.parse_args()

    n_rep = 10 if args.quick else args.n_rep
    kA = args.D / 2.0
    dk = 0.3 * np.sqrt(args.D)
    tasks = [(rep, args.N, args.D, kA, dk, args.n_perm, args.include_coupling)
             for rep in range(n_rep)]

    print(f'# synth_radial_control | D={args.D} N={args.N} kA={kA:.0f} '
          f'dk={dk:.2f} n_perm={args.n_perm} reps={n_rep} '
          f'M={min(2000, args.N // 3)} k=60 alpha={ALPHA}', flush=True)

    all_rows = []
    if args.jobs > 1:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            for i, (rows, dt) in enumerate(pool.imap_unordered(run_rep, tasks)):
                all_rows += rows
                print(f'  rep done ({i + 1}/{n_rep}, {dt:.0f}s)', flush=True)
    else:
        for i, task in enumerate(tasks):
            rows, dt = run_rep(task)
            all_rows += rows
            print(f'  rep {task[0]} done ({i + 1}/{n_rep}, {dt:.0f}s)', flush=True)

    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cond', 'test', 'rep', 'p'])
        w.writerows(all_rows)

    # ── summary ──
    conds = sorted({r[0] for r in all_rows})
    tests = ['VRT', 'NORM-MED', 'NORM-DISP']
    print('\n| condition | ' + ' | '.join(f'{t} rej@5% (med p)' for t in tests) + ' |')
    print('|---' * (len(tests) + 1) + '|')
    for c in conds:
        cells = []
        for t in tests:
            ps = np.array([r[3] for r in all_rows if r[0] == c and r[1] == t])
            cells.append(f'{(ps < ALPHA).sum()}/{len(ps)} ({np.median(ps):.3f})')
        print(f'| {c} | ' + ' | '.join(cells) + ' |')

    # ── S3 vs S4 invariance ──
    p3 = {r[2]: r[3] for r in all_rows if r[0] == 'S3_angular' and r[1] == 'VRT'}
    p4 = {r[2]: r[3] for r in all_rows
          if r[0] == 'S4_angular_plus_radial' and r[1] == 'VRT'}
    if p3 and p4:
        dp = np.array([abs(p3[k] - p4[k]) for k in sorted(p3) if k in p4])
        print(f'\nInvariance check S3 vs S4 (same directions, radii x{RAD_SCALE}): '
              f'max |dp| = {dp.max():.2e}, median |dp| = {np.median(dp):.2e} '
              f'over {len(dp)} reps')
    print('\nDone. CSV:', args.out)


if __name__ == '__main__':
    sys.exit(main())
