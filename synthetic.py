"""Synthetic controls: Section 4.4 / Appendix D and reference choice in Appendix G."""
import argparse
import csv
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from vrt import vrt_full, perm_p
from baselines import mmd_test_fast, median_sigma, rbf_matrix, mmd2_from_matrix

D, N = 500, 5000
KB, SQD = D * 0.5, np.sqrt(D)
TEST_SEED = 10 ** 6
KAPPA0 = 2500.0
LOGMU, SIG0, SIG_TAIL, RAD_SCALE = np.log(10.0), 0.05, 0.20, 1.03
RADIAL_DEFAULT = ("S0_null", "S1_radial_mean", "S2_radial_tail", "S3_angular", "S4_angular_plus_radial")
VRT_TIER = {5000: (2000, 60), 1500: (500, 15), 500: (166, 5)}

def sample_uniform(n, D, rng):
    x = rng.standard_normal((n, D))
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def radial_vmf(n, D, mu, kappa, rng):
    if kappa < 1e-10:
        return sample_uniform(n, D, rng)
    d = D
    b = (-2 * kappa + np.sqrt(4 * kappa ** 2 + (d - 1) ** 2)) / (d - 1)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * np.log(1 - x0 ** 2)
    samples = np.empty((n, D))
    idx = 0
    batch = max(n, 256)
    while idx < n:
        z = rng.beta((d - 1) / 2, (d - 1) / 2, size=batch)
        w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        u = rng.uniform(size=batch)
        accept = kappa * w + (d - 1) * np.log(1 - x0 * w) - c >= np.log(u)
        ws = w[accept]
        needed = min(len(ws), n - idx)
        v = rng.standard_normal((needed, D - 1))
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        samples[idx:idx + needed, 0] = ws[:needed]
        samples[idx:idx + needed, 1:] = v * np.sqrt(1 - ws[:needed] ** 2)[:, None]
        idx += needed
    e1 = np.zeros(D)
    e1[0] = 1.0
    if np.allclose(mu, e1):
        return samples
    u_vec = e1 + mu
    u_vec = u_vec / np.linalg.norm(u_vec)
    return samples - 2 * np.outer(samples @ u_vec, u_vec) + 2 * np.outer(samples @ e1, mu)

def normalize(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def build_conditions(rng, N, D, kA, kB, include_coupling):
    """Return dict: name -> (directions, radii) for sample B."""
    conds = {}
    u = radial_vmf(N, D, _e1(D), kA, rng)
    r = np.exp(rng.normal(LOGMU, SIG0, N))
    conds['S0_null'] = (u, r)
    u = radial_vmf(N, D, _e1(D), kA, rng)
    r = np.exp(rng.normal(LOGMU, SIG0, N)) * RAD_SCALE
    conds['S1_radial_mean'] = (u, r)
    u = radial_vmf(N, D, _e1(D), kA, rng)
    r = np.exp(rng.normal(LOGMU, SIG_TAIL, N))
    conds['S2_radial_tail'] = (u, r)
    u34 = radial_vmf(N, D, _e1(D), kB, rng)
    r34 = np.exp(rng.normal(LOGMU, SIG0, N))
    conds['S3_angular'] = (u34, r34)
    conds['S4_angular_plus_radial'] = (u34, r34 * RAD_SCALE)
    if include_coupling:
        u5 = radial_vmf(N, D, _e1(D), kA, rng)
        r5 = np.exp(rng.normal(LOGMU, SIG0, N))
        r5c = np.empty(N)
        r5c[np.argsort(u5[:, 0])] = np.sort(r5)
        conds['S5_coupling'] = (u5, r5c)
    return conds

def _e1(D):
    e1 = np.zeros(D)
    e1[0] = 1.0
    return e1

def median_perm_test(a, b, n_perm, rng):
    return perm_p(a, b, n_perm, rng)[1]

def sample_vmf(n, D, kappa, rng):
    mu = np.zeros(D)
    mu[0] = 1.0
    d = D
    b = (-2 * kappa + np.sqrt(4 * kappa ** 2 + (d - 1) ** 2)) / (d - 1)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * np.log(1 - x0 ** 2)
    samples = np.empty((n, D))
    idx = 0
    while idx < n:
        z = rng.beta((d - 1) / 2, (d - 1) / 2, size=n)
        w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        u = rng.uniform(size=n)
        accept = kappa * w + (d - 1) * np.log(1 - x0 * w) - c >= np.log(u)
        ws = w[accept]
        need = min(len(ws), n - idx)
        v = rng.standard_normal((need, D - 1))
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        sq = np.sqrt(np.maximum(1 - ws[:need] ** 2, 0))[:, None]
        samples[idx:idx + need] = np.concatenate([ws[:need, None], sq * v], axis=1)
        idx += need
    return samples

def sample_vmf_rotated(n, D, kappa, mu2, rng):
    samples = sample_vmf(n, D, kappa, rng)
    e1 = np.zeros(D)
    e1[0] = 1.0
    if np.allclose(mu2, e1):
        return samples
    if np.allclose(mu2, -e1):
        samples[:, 0] *= -1
        return samples
    u_vec = mu2 - e1
    u_vec = u_vec / np.linalg.norm(u_vec)
    return samples - 2 * np.outer(samples @ u_vec, u_vec)

def sample_vmf_mixture(n, D, kappa_base, eps, kappa_spike, rng):
    mu = np.zeros(D)
    mu[0] = 1.0
    n_spike = rng.binomial(n, eps)
    n_base = n - n_spike
    parts = []
    if n_base > 0:
        parts.append(sample_vmf(n_base, D, kappa_base, rng))
    if n_spike > 0:
        parts.append(sample_vmf(n_spike, D, kappa_spike, rng))
    out = np.concatenate(parts, 0)
    return out[rng.permutation(n)]



CONDS = [(10, 'null', 'null', 0.0), (10, 'tight_0.05', 'tight', 0.05), (10, 'memo_0.10', 'memo', 0.1), (10, 'shift_4', 'shift', 4.0), (10, 'tight_0.03', 'tight', 0.03), (10, 'tight_0.07', 'tight', 0.07), (10, 'tight_0.10', 'tight', 0.1), (10, 'tight_0.30', 'tight', 0.3), (100, 'null', 'null', 0.0), (100, 'tight_0.05', 'tight', 0.05), (100, 'memo_0.10', 'memo', 0.1), (100, 'shift_4', 'shift', 4.0)]

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
    if kind == 'tight':
        kA[:] = KAPPA0 * (1 + param)
    elif kind == 'memo':
        kA[:int(round(param * N))] = 3 * KAPPA0
    elif kind == 'shift':
        cB = rotate_small(centres, param, rng)
    A = cluster_sample(N, centres, kA, rng)
    A = A[rng.permutation(N)]
    B = cluster_sample(N, cB, kB, rngB)
    return (A, B)

def vrt_setting(n):
    if n in VRT_TIER:
        return VRT_TIER[n]
    M = max(n // 3, 20)
    return (M, max(int(round(0.03 * M)), 2))

def single_cluster_rep(args):
    rep, budgets, n_perm = args
    t0 = time.time()
    ts = TEST_SEED + rep
    rows = []
    sub = {}
    for n in budgets:
        sub[n] = (np.random.RandomState(ts + 7).permutation(N)[:n], np.random.RandomState(ts + 8).permutation(N)[:n])

    def both(name, P, Q, norms=None):
        for n in budgets:
            iP, iQ = sub[n]
            Pn, Qn = (P, Q) if n >= N else (P[iP], Q[iQ])
            M, k = vrt_setting(n)
            _, pv = vrt_full(Pn, Qn, D=D, k=k, M=M, n_qtr=n - M, n_perm=n_perm, seed=ts)
            _, pm = mmd_test_fast(Pn, Qn, n_sub=n, n_perm=n_perm, seed=ts, backend='numpy')
            rows.extend([(name, 'VRT', n, rep, pv), (name, 'MMD', n, rep, pm)])
            if norms is not None:
                nA, nB = norms
                nA, nB = (nA, nB) if n >= N else (nA[iP], nB[iQ])
                p_nm = median_perm_test(nA, nB, 500, np.random.RandomState(ts))
                m = np.median(np.log(np.concatenate([nA, nB])))
                p_nd = median_perm_test(np.abs(np.log(nA) - m), np.abs(np.log(nB) - m), 500, np.random.RandomState(ts + 1))
                rows.extend([(name, 'NORM-MED', n, rep, p_nm), (name, 'NORM-DISP', n, rep, p_nd)])
    P = sample_vmf(N, D, KB, np.random.RandomState(rep))
    for name, qfn in ANGULAR:
        Q = qfn(np.random.RandomState(rep + 1000))
        both(name, P, Q)
    rng = np.random.RandomState(20260724 + rep)
    kA = D / 2.0
    kB = kA + 0.3 * np.sqrt(D)
    uA = radial_vmf(N, D, _e1(D), kA, rng)
    rA = np.exp(rng.normal(LOGMU, SIG0, N))
    A = rA[:, None] * uA
    A_norm = normalize(A)
    normsA = np.linalg.norm(A, axis=1)
    conds = build_conditions(rng, N, D, kA, kB, include_coupling=True)
    for name in RADIAL:
        u, r = conds[name]
        B = r[:, None] * u
        both(name, A_norm, normalize(B), norms=(normsA, np.linalg.norm(B, axis=1)))
    return (rows, time.time() - t0)

def cluster_rep(args):
    rep, n_perm, conds = args
    t0 = time.time()
    ts = TEST_SEED + rep
    rows = []
    for C, name, kind, param in conds:
        A, B = make(C, kind, param, rep)
        _, pv = vrt_full(A, B, D=D, k=vrt_setting(N)[1], M=vrt_setting(N)[0], n_qtr=N - vrt_setting(N)[0], n_perm=n_perm, seed=ts)
        _, pm = mmd_test_fast(A, B, n_sub=N, n_perm=n_perm, seed=ts, backend='numpy')
        rows += [(name, C, 'VRT', rep, pv), (name, C, 'MMD', rep, pm)]
    return (rows, time.time() - t0)

def mmd_scales(A, B, scales, n_perm, seed):
    """p-values of the permutation MMD at bandwidths s * sigma_med, same subsample order, bandwidth draw and permutations"""
    rng = np.random.RandomState(seed)
    X = A[rng.permutation(len(A))]
    Y = B[rng.permutation(len(B))]
    pooled = np.concatenate([X, Y])
    sigma = median_sigma(pooled, rng)
    n, m = (len(X), len(Y))
    Nn = n + m
    perms = np.stack([rng.permutation(Nn) for _ in range(n_perm)])
    P = np.zeros((Nn, n_perm + 1), dtype=np.float64)
    P[:n, 0] = 1.0
    P[perms[:, :n].T, np.arange(1, n_perm + 1)[None, :]] = 1.0
    out = {}
    for s in scales:
        K = rbf_matrix(pooled, sigma * s, 'numpy')
        val = mmd2_from_matrix(K, P, n, m)
        out[s] = float((np.sum(val[1:] >= val[0]) + 1) / (n_perm + 1))
    return (out, sigma)

def class_distances(A, rep, C, n_sub=1500):
    """median Euclidean distance within and between classes in sample A (classes = nearest centre)"""
    rng = np.random.RandomState(rep * 1000 + C)
    centres = uniform_dirs(C, rng)
    sub = A[np.random.RandomState(rep).permutation(len(A))[:n_sub]]
    lab = np.argmax(sub @ centres.T, axis=1)
    sq = np.sum(sub ** 2, 1, keepdims=True)
    d = np.sqrt(np.maximum(sq + sq.T - 2 * sub @ sub.T, 0.0))
    same = lab[:, None] == lab[None, :]
    iu = np.triu_indices(len(sub), 1)
    return (float(np.median(d[iu][same[iu]])), float(np.median(d[iu][~same[iu]])))

def bandwidth_rep(args):
    rep, scales, n_perm, conds = args
    t0 = time.time()
    ts = TEST_SEED + rep
    rows = []
    for C, name, kind, param in conds:
        A, B = make(C, kind, param, rep)
        dw, db = class_distances(A, rep, C)
        pm, sigma = mmd_scales(A, B, scales, n_perm, ts)
        _, pv = vrt_full(A, B, D=D, k=vrt_setting(N)[1], M=vrt_setting(N)[0], n_qtr=N - vrt_setting(N)[0], n_perm=n_perm, seed=ts)
        rows.append((name, 0.0, 'VRT', rep, pv, sigma, dw, db))
        for s in scales:
            rows.append((name, s, 'MMD', rep, pm[s], sigma, dw, db))
    return (rows, time.time() - t0)

def reference_vmf(n, D, mu, kappa, rng):
    if kappa < 1e-10:
        return sample_uniform(n, D, rng)
    d = D
    b = (-2 * kappa + np.sqrt(4 * kappa ** 2 + (d - 1) ** 2)) / (d - 1)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * np.log(1 - x0 ** 2)
    samples = np.empty((n, D))
    idx = 0
    batch = max(n, 256)
    while idx < n:
        z = rng.beta((d - 1) / 2, (d - 1) / 2, size=batch)
        w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        u = rng.uniform(size=batch)
        accept = kappa * w + (d - 1) * np.log(1 - x0 * w) - c >= np.log(u)
        w = w[accept]
        if len(w) == 0:
            continue
        need = min(len(w), n - idx)
        w = w[:need]
        v = rng.standard_normal((need, D - 1))
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        sq = np.sqrt(np.maximum(1 - w ** 2, 0))[:, None]
        pts = np.concatenate([w[:, None], sq * v], axis=1)
        e1 = np.zeros(D)
        e1[0] = 1.0
        if not np.allclose(mu, e1):
            u_vec = mu - e1
            u_vec = u_vec / np.linalg.norm(u_vec)
            pts = pts - 2 * np.outer(pts @ u_vec, u_vec)
        samples[idx:idx + need] = pts
        idx += need
    return samples

def reference_mixture(n, D, eps, kappa, rng):
    mu_mix = np.zeros(D)
    mu_mix[0] = 1.0
    n_cluster = rng.binomial(n, eps)
    n_uniform = n - n_cluster
    parts = []
    if n_uniform > 0:
        parts.append(sample_uniform(n_uniform, D, rng))
    if n_cluster > 0:
        parts.append(reference_vmf(n_cluster, D, mu_mix, kappa, rng))
    out = np.concatenate(parts, 0)
    rng.shuffle(out)
    return out


REFERENCE_CONDITIONS = ("conc_0.3", "mean_2deg", "mixture_0.05", "unif_vs_vmf")


def reference_samples(condition, rep):
    """Appendix G draws A then B from one stream, restarting it for each condition."""
    rng = np.random.RandomState(rep)
    mu = _e1(D)
    if condition == "conc_0.3":
        a = reference_vmf(N, D, mu, KB, rng)
        b = reference_vmf(N, D, mu, KB + 0.3 * np.sqrt(D), rng)
    elif condition == "mean_2deg":
        a = reference_vmf(N, D, mu, KB, rng)
        b = reference_vmf(N, D, MU2, KB, rng)
    elif condition == "mixture_0.05":
        a = sample_uniform(N, D, rng)
        b = reference_mixture(N, D, 0.05, D * 2.0, rng)
    elif condition == "unif_vs_vmf":
        a = sample_uniform(N, D, rng)
        b = reference_vmf(N, D, mu, KB, rng)
    else:
        raise ValueError(f"Unknown reference condition: {condition}")
    return a.astype(np.float64), b.astype(np.float64)


def reference_rep(args):
    rep, n_perm, conditions = args
    start = time.time()
    rows = []
    reference, neighbours = vrt_setting(N)
    for name in conditions:
        a, b = reference_samples(name, rep)
        for direction, first, second in (("A", a, b), ("B", b, a)):
            statistic, p = vrt_full(first, second, D, M=reference, k=neighbours,
                                    n_perm=n_perm, seed=rep)
            rows.append((name, direction, rep, statistic, p))
    return rows, time.time() - start

def configure(points=5000, dimension=500):
    global N, D, KB, SQD, MU2, ANGULAR, RADIAL
    if points < 1 or dimension < 2:
        raise ValueError("Need positive point count and dimension at least two")
    N, D = points, dimension
    KB, SQD = D * 0.5, np.sqrt(D)
    MU2 = np.zeros(D)
    MU2[0], MU2[1] = np.cos(np.radians(2.0)), np.sin(np.radians(2.0))
    RADIAL = list(RADIAL_DEFAULT)
    ANGULAR = [
        ("null_vmf", lambda rng: sample_vmf(N, D, KB, rng)),
        ("conc_0.3", lambda rng: sample_vmf(N, D, KB + 0.3 * SQD, rng)),
        ("mean_2deg", lambda rng: sample_vmf_rotated(N, D, KB, MU2, rng)),
        ("mix_0.05", lambda rng: sample_vmf_mixture(N, D, KB, 0.05, KB * 3, rng)),
    ]


def configure_worker(points, dimension, selected, experiment):
    """Restore the requested experiment in each spawned worker."""
    configure(points, dimension)
    if selected and experiment == "single-cluster":
        global ANGULAR, RADIAL
        ANGULAR = [(name, fn) for name, fn in ANGULAR if name in selected]
        RADIAL = [name for name in RADIAL if name in selected]


configure()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("clusters", "single-cluster", "bandwidth", "reference"))
    parser.add_argument("--reps", type=int, default=None, help="200 by default; 500 for reference")
    parser.add_argument("--rep-offset", type=int, default=None)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--budgets", default="5000,1500,500")
    parser.add_argument("--conditions", default="", help="Comma-separated condition names; default all")
    parser.add_argument("--scales", default="0.25,0.5,1,2")
    parser.add_argument("--smoke", action="store_true", help="One repetition, 300 points, D=32, 19 permutations")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.reps is None:
        args.reps = 500 if args.experiment == "reference" else 200
    if args.reps < 1 or args.permutations < 1 or args.jobs < 1:
        parser.error("reps, permutations and jobs must be positive")
    configure(300, 32) if args.smoke else configure()
    reps, permutations = (1, 19) if args.smoke else (args.reps, args.permutations)
    offset = args.rep_offset if args.rep_offset is not None else (200 if args.experiment == "bandwidth" else 0)
    repetitions = range(offset, offset + reps)
    selected = {name.strip() for name in args.conditions.split(",")} if args.conditions else None
    if selected and args.experiment in ("clusters", "bandwidth"):
        available = {condition[1] for condition in CONDS}
        if not selected <= available:
            parser.error("Unknown cluster condition: " + ", ".join(sorted(selected - available)))
    conditions = [c for c in CONDS if selected is None or c[1] in selected]
    if args.conditions and not conditions and args.experiment in ("clusters", "bandwidth"):
        parser.error("No matching cluster conditions")
    if args.experiment == "clusters":
        function = cluster_rep
        tasks = [(rep, permutations, conditions) for rep in repetitions]
        fields = ["cond", "C", "test", "rep", "p"]
    elif args.experiment == "single-cluster":
        budgets = [300] if args.smoke else sorted({int(n) for n in args.budgets.split(",")}, reverse=True)
        if any(n < 100 or n > N for n in budgets):
            parser.error(f"budgets must lie in [100, {N}]")
        if selected:
            global ANGULAR, RADIAL
            available = {name for name, _ in ANGULAR} | set(RADIAL)
            if not selected <= available:
                parser.error("Unknown single-cluster condition")
            ANGULAR = [(name, fn) for name, fn in ANGULAR if name in selected]
            RADIAL = [name for name in RADIAL if name in selected]
        function = single_cluster_rep
        tasks = [(rep, budgets, permutations) for rep in repetitions]
        fields = ["cond", "test", "budget", "rep", "p"]
    elif args.experiment == "reference":
        if selected and not selected <= set(REFERENCE_CONDITIONS):
            parser.error("Unknown reference condition")
        names = [name for name in REFERENCE_CONDITIONS if selected is None or name in selected]
        function = reference_rep
        tasks = [(rep, permutations, names) for rep in repetitions]
        fields = ["cond", "reference", "rep", "statistic", "p"]
    else:
        chosen = selected or {"null", "tight_0.05"}
        conditions = [c for c in CONDS if c[0] == 10 and c[1] in chosen]
        scales = [float(s) for s in args.scales.split(",")]
        if not scales or any(not np.isfinite(s) or s <= 0 for s in scales):
            parser.error("bandwidth scales must be positive finite numbers")
        function = bandwidth_rep
        tasks = [(rep, scales, permutations, conditions) for rep in repetitions]
        fields = ["cond", "scale", "test", "rep", "p", "sigma", "d_within", "d_between"]
    path = Path(args.out)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.jobs > 1:
        with Pool(args.jobs, initializer=configure_worker,
                  initargs=(N, D, selected, args.experiment)) as pool:
            for chunk, duration in pool.imap_unordered(function, tasks):
                rows.extend(chunk)
                print(f"repetition completed in {duration:.2f}s", flush=True)
    else:
        for task in tasks:
            chunk, duration = function(task)
            rows.extend(chunk)
            print(f"rep={task[0]} completed in {duration:.2f}s", flush=True)
    if not all(np.isfinite(row[fields.index("p")]) and 0 <= row[fields.index("p")] <= 1 for row in rows):
        raise FloatingPointError("Invalid synthetic p-value")
    with path.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(sorted(rows, key=lambda row: tuple(map(str, row))))
    print(f"Wrote {len(rows)} records to {path}" + (" (SMOKE, not paper results)" if args.smoke else ""))


if __name__ == "__main__":
    main()
