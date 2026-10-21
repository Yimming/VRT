"""
mmd_tst_fast.py — Gaussian-RBF MMD^2 (unbiased) permutation test on precomputed per-epoch features.
Same test as methods_v2.mmd_test (paper Sec. 4 baseline), exact but fast.

Reproduces the original function step for step, with the SAME RandomState stream:
  rng = RandomState(seed)
  X = emb_tr[rng.permutation(len)[:n_sub]];  Y = emb_te[rng.permutation(len)[:n_sub]]
  bandwidth: median heuristic on rng.choice(pooled, min(2000, N), replace=False), sigma = sqrt(median d^2 > 0)
  statistic: unbiased MMD^2 = sum Kxx/(n(n-1)) + sum Kyy/(m(m-1)) - 2 sum Kxy/(nm), diagonals excluded
  permutations: rng.permutation(N), first n -> X, n_perm times;  p = (#{>= obs} + 1)/(n_perm + 1)
The only change is HOW the permuted statistics are evaluated: the pooled kernel matrix is computed once
(float64, zero diagonal) and every permuted MMD^2 is a quadratic form in it, all permutations in one
matrix product (same device as ladder_stats.mmd_cached, but vectorised). --selftest checks against a
verbatim copy of methods_v2.mmd_test: identical p-values, statistic equal to 1e-10 (float64 inputs).

Usage:
  python mmd_tst_fast.py --selftest
  python mmd_tst_fast.py --tag r50 --npz_fmt ".../r50/ep{ep:03d}.npz" --n_samples 1500 --n_perm 2000 \
      --seeds 42-51 --epochs 0-1000 --backend torch --out mmd_r50_n1500.csv
  (usage example of the earlier stage; the published 50-seed runs used --seeds 42-91 and --n_perm 2000, see tasks/)
"""
import os, sys, csv, time, argparse
import numpy as np
from release_utils import validate_features, check_statistics, ResumeIndex, open_append


# ----------------------------------------------------------------------------- kernel matrix
def median_sigma(pooled, rng):
    """Median heuristic exactly as methods_v2.mmd_test (rng.choice on the pooled sample, <= 2000 points)."""
    sub = pooled[rng.choice(len(pooled), min(2000, len(pooled)), replace=False)].astype(np.float64)
    sq = np.sum(sub ** 2, 1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2 * sub @ sub.T, 0.0)
    np.fill_diagonal(d2, np.inf)
    vals = d2[np.isfinite(d2) & (d2 > 0)]
    return max(float(np.sqrt(np.median(vals))), 1e-8) if len(vals) > 0 else 1.0


def rbf_matrix(Z, sigma, backend="torch"):
    """Pooled RBF kernel matrix exp(-d^2 / (2 sigma^2)), float64, zero diagonal.
    torch backend keeps it on the GPU when available (memory ~ 8 N^2 bytes, 3.2 GB at N = 20,000)."""
    if backend == "torch":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        T = torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float64)).to(dev)
        sq = (T * T).sum(1)
        K = T @ T.T
        K.mul_(-2.0).add_(sq[:, None]).add_(sq[None, :])      # squared distances, in place
        K.clamp_(min=0.0).mul_(-1.0 / (2 * sigma ** 2)).exp_()
        K.fill_diagonal_(0.0)
        return K
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    sq = (Z * Z).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T), 0.0)
    K = np.exp(-d2 / (2 * sigma ** 2))
    np.fill_diagonal(K, 0.0)
    return K


# ----------------------------------------------------------------------------- statistic
def mmd2_from_matrix(K, P, n, m):
    """Unbiased MMD^2 for every labelling column of P (N x B, 0/1, column sums n)."""
    is_torch = not isinstance(K, np.ndarray)
    if is_torch:
        import torch
        P = torch.as_tensor(P, dtype=K.dtype, device=K.device)
    else:
        P = np.asarray(P, dtype=np.float64)
    tot = K.sum()                         # 1' K 1  (diagonal already zero)
    KP = K @ P                            # (N,B)
    S1all = KP.sum(0)                     # 1' K 1_X
    Sxx = (P * KP).sum(0)                 # 1_X' K 1_X  = sum Kxx off-diagonal
    Syy = tot - 2.0 * S1all + Sxx         # 1_Y' K 1_Y
    Sxy = S1all - Sxx                     # 1_X' K 1_Y
    val = Sxx / (n * (n - 1)) + Syy / (m * (m - 1)) - 2.0 * Sxy / (n * m)
    return val.cpu().numpy() if is_torch else val


def mmd_test_fast(emb_tr, emb_te, n_sub=1500, n_perm=2000, seed=42, backend="torch"):
    """Drop-in replacement for methods_v2.mmd_test: same signature, same draws, same output (obs, p)."""
    rng = np.random.RandomState(seed)
    n_x = min(n_sub, len(emb_tr)); n_y = min(n_sub, len(emb_te))
    X = emb_tr[rng.permutation(len(emb_tr))[:n_x]]
    Y = emb_te[rng.permutation(len(emb_te))[:n_y]]
    pooled = np.concatenate([X, Y])
    if not np.isfinite(pooled).all():                       # release fix: a NaN/inf input used to end as p = 1/(B+1)
        raise ValueError("mmd_test_fast: the subsample contains NaN or inf")
    sigma = median_sigma(pooled, rng)                       # consumes the same rng.choice draw
    N, n, m = len(pooled), n_x, n_y
    K = rbf_matrix(pooled, sigma, backend)
    perms = np.stack([rng.permutation(N) for _ in range(n_perm)])          # same draws as the original loop
    P = np.zeros((N, n_perm + 1), dtype=np.float64)
    P[:n, 0] = 1.0
    P[perms[:, :n].T, np.arange(1, n_perm + 1)[None, :]] = 1.0
    val = mmd2_from_matrix(K, P, n, m)
    obs, perm_vals = float(val[0]), val[1:]
    check_statistics(obs, perm_vals, "MMD^2")               # release fix: never compare against a NaN observation
    p = float((np.sum(perm_vals >= obs) + 1) / (n_perm + 1))
    return obs, p


# ----------------------------------------------------------------------------- reference (verbatim)
def mmd_test_original(emb_tr, emb_te, n_sub=1500, n_perm=1000, seed=42):
    """VERBATIM copy of methods_v2.mmd_test, for the self-test only."""
    rng = np.random.RandomState(seed)
    n_x = min(n_sub, len(emb_tr))
    n_y = min(n_sub, len(emb_te))
    X = emb_tr[rng.permutation(len(emb_tr))[:n_x]]
    Y = emb_te[rng.permutation(len(emb_te))[:n_y]]
    # Median heuristic (Gretton et al. 2012)
    pooled = np.concatenate([X, Y])
    sub = pooled[rng.choice(len(pooled), min(2000, len(pooled)), replace=False)]
    d2 = np.sum(sub**2,1,keepdims=True) + np.sum(sub**2,1,keepdims=True).T - 2*sub@sub.T
    d2 = np.maximum(d2, 0.0)
    np.fill_diagonal(d2, np.inf)
    vals = d2[np.isfinite(d2) & (d2 > 0)]
    sigma = max(float(np.sqrt(np.median(vals))), 1e-8) if len(vals) > 0 else 1.0
    def rbf(A, B, s):
        d2 = np.sum(A**2,1,keepdims=True)+np.sum(B**2,1,keepdims=True).T-2*A@B.T
        d2 = np.maximum(d2, 0.0)
        return np.exp(-d2/(2*s**2))
    def mmd2(X, Y, s):
        n, m = len(X), len(Y)
        Kxx = rbf(X,X,s); np.fill_diagonal(Kxx,0)
        Kyy = rbf(Y,Y,s); np.fill_diagonal(Kyy,0)
        Kxy = rbf(X,Y,s)
        return float(Kxx.sum()/(n*(n-1))+Kyy.sum()/(m*(m-1))-2*Kxy.sum()/(n*m))
    obs = mmd2(X, Y, sigma)
    pooled_all = np.concatenate([X, Y]); n = len(X)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled_all))
        if mmd2(pooled_all[p[:n]], pooled_all[p[n:]], sigma) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def selftest():
    ok = True
    print("fast vs verbatim methods_v2.mmd_test (same seed => same subsample, bandwidth draw and permutations)")
    cases = [(300, 200, 16, 0.0, 60, 100), (250, 250, 64, 0.15, 80, 150), (400, 300, 2048, 0.03, 50, 100),
             (120, 90, 8, 0.0, 200, 60)]                      # last: n_sub > len(emb_te) -> unequal n, m
    for trial, (n_tr, n_te, D, shift, n_sub, n_perm) in enumerate(cases):
        rng = np.random.default_rng(trial)
        tr = rng.standard_normal((n_tr, D)); te = rng.standard_normal((n_te, D)) + shift
        tr /= np.linalg.norm(tr, axis=1, keepdims=True); te /= np.linalg.norm(te, axis=1, keepdims=True)
        for dtype in (np.float64, np.float32):
            a, b = tr.astype(dtype), te.astype(dtype)
            for seed in (42, 43):
                o_obs, o_p = mmd_test_original(a, b, n_sub, n_perm, seed)
                for backend in ("numpy", "torch"):
                    try:
                        f_obs, f_p = mmd_test_fast(a, b, n_sub, n_perm, seed, backend)
                    except ImportError:
                        continue
                    # float64: relative agreement; float32: the ORIGINAL evaluates the kernel in float32, so its
                    # MMD^2 (a small difference of O(1) kernel means) carries ~1e-7 absolute rounding error
                    err = abs(f_obs - o_obs) / max(abs(o_obs), 1e-12)
                    good = (f_p == o_p) and ((err < 1e-9) if dtype is np.float64 else (abs(f_obs - o_obs) < 2e-6))
                    ok &= good
                    print(f"  n_tr={n_tr} n_te={n_te} D={D:4d} shift={shift} n_sub={n_sub} B={n_perm} seed={seed} "
                          f"{np.dtype(dtype).name} {backend:6s}  obs {f_obs:+.6e} vs {o_obs:+.6e} (rel {err:.1e})  "
                          f"p {f_p:.4f} vs {o_p:.4f}  {'OK' if good else 'MISMATCH'}")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


# ----------------------------------------------------------------------------- I/O helpers
KEY_PAIRS = [("h_tr", "h_te"), ("X_tr", "X_te"), ("tr", "te"), ("train", "test"), ("emb_tr", "emb_te")]


def load_epoch(fmt, ep):
    f = fmt.format(ep=ep)
    if not os.path.exists(f):
        return None
    d = np.load(f)
    for a, b in KEY_PAIRS:
        if a in d.files and b in d.files:
            return d[a], d[b]
    raise KeyError(f"unknown keys in {f}: {d.files}")


def l2n(x):
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def parse_range(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tag")
    ap.add_argument("--npz_fmt")
    ap.add_argument("--n_samples", type=int, default=1500)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seeds", default="42-51")
    ap.add_argument("--epochs", default="0-1000")
    ap.add_argument("--raw", action="store_true", help="skip L2 normalization")
    ap.add_argument("--backend", choices=["torch", "numpy"], default="torch")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not (args.tag and args.npz_fmt and args.out):
        ap.error("--tag, --npz_fmt and --out are required (or use --selftest)")

    seeds = parse_range(args.seeds); epochs = parse_range(args.epochs)
    # release fix: resume by (epoch, seed), counting only valid rows of this run (tag, sample budget, permutation count,
    # feature normalization); rows of another run make the resume refuse (one output file holds one run). The runner
    # also refuses a budget larger than a split (below), so the n_samples column always equals the requested budget.
    # The as-run files have no "features" column; a completed as-run file is not resumed, a new file is written.
    features = "raw" if args.raw else "l2"
    HEADER = ["tag", "n_samples", "n_perm", "epoch", "seed", "mmd2", "p", "features"]
    index = ResumeIndex(args.out, key_cols=("epoch", "seed"), value_cols=("mmd2", "p"), ranges={"p": (0, 1)},
                        expect={"tag": args.tag, "n_samples": args.n_samples, "n_perm": args.n_perm, "features": features},
                        header=HEADER)
    print(index.report())
    fout, w = open_append(args.out, HEADER)

    print(f"[{args.tag}] MMD (RBF, median heuristic, unbiased), n_sub={args.n_samples}/split, perms={args.n_perm}, "
          f"seeds={seeds[0]}-{seeds[-1]}, {'raw' if args.raw else 'L2-normalized'} features, backend={args.backend}", flush=True)
    for ep in epochs:
        todo = [s for s in seeds if not index.has(ep, s)]
        if not todo:
            continue
        loaded = load_epoch(args.npz_fmt, ep)
        if loaded is None:
            continue
        t0 = time.time()
        e_tr, e_te = validate_features(*loaded, names=("train features", "test features"), allow_zero_rows=args.raw)
        if min(len(e_tr), len(e_te)) < args.n_samples:       # release fix: never silently test a smaller budget
            raise SystemExit(f"ep {ep}: split sizes {len(e_tr)}/{len(e_te)} are below --n_samples {args.n_samples}")
        if not args.raw:                                     # a zero row is a valid point of the raw feature space
            e_tr, e_te = l2n(e_tr), l2n(e_te)
        ps = []
        for seed in todo:
            obs, p = mmd_test_fast(e_tr, e_te, args.n_samples, args.n_perm, seed, args.backend)
            w.writerow([args.tag, args.n_samples, args.n_perm, ep, seed, f"{obs:.8e}", f"{p:.6f}", features])
            ps.append(p)
        fout.flush()
        print(f"  ep{ep:4d}  med_p={np.median(ps):.4f}  reject {sum(p < 0.05 for p in ps)}/{len(ps)}  "
              f"first seed p={ps[0]:.4f}  ({time.time() - t0:.1f} s for {len(todo)} seeds)", flush=True)
    fout.close(); print("done")


if __name__ == "__main__":
    main()
