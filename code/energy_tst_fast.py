"""
energy_tst_fast.py — Energy Distance two-sample permutation test on precomputed per-epoch features.
Same test as methods_v2.energy_distance_test (paper Sec. 4 baseline), exact but fast.

Reproduces the original function step for step:
  rng = np.random.RandomState(seed); X = emb_tr[rng.permutation(len)[:n_sub]]; Y = emb_te[...]   (same draws)
  statistic  E = 2 mean||x-y|| - mean||x-x'|| - mean||y-y'||   (V form, all n^2 pairs incl. zero diagonal)
  permutations p = rng.permutation(2n) from the SAME rng stream, first n -> sample 1, n_perm times
  p-value  (#{E_perm >= E_obs} + 1)/(n_perm + 1)
so the subsample, the 500 permutations and the decision rule are identical to the original. The only
change is HOW E is evaluated. The original recomputes every pairwise distance for every permutation
(500 x (2n)^2 x D flops, about 32 h per checkpoint at D=2,048). A permutation only relabels the pooled
points, so the pooled distance matrix is fixed; it is computed once per subsample (2n x 2n, float64)
and every permuted E is a quadratic form in it, evaluated for all permutations with one matrix product.
--selftest checks against a verbatim copy of the original function (identical p-values, E to 1e-10)
and against brute-force recomputation with the same permutations, on equal and unequal sizes.

Usage:
  python energy_tst_fast.py --selftest
  python energy_tst_fast.py --tag r50 --npz_fmt ".../r50/ep{ep:03d}.npz" --n_samples 1500 \
      --n_perm 500 --seeds 42-51 --epochs 0-1000 --backend numpy --out ed_r50_n1500.csv
  (usage example of the earlier stage; the published 50-seed runs used --seeds 42-91 and --n_perm 2000, see tasks/)
"""
import os, sys, csv, time, argparse
import numpy as np
from release_utils import validate_features, check_statistics, ResumeIndex, open_append


# ----------------------------------------------------------------------------- distance matrix
def pairwise_dist(Z, backend="numpy"):
    """Exact Euclidean distance matrix of the pooled sample, float64, symmetric, zero diagonal.
    Returns a torch tensor (GPU if available) for backend='torch', else a numpy array."""
    if backend == "torch":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        T = torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float64)).to(dev)
        sq = (T * T).sum(1)
        d = T @ T.T                                              # Gram matrix, then squared distances in place
        d.mul_(-2.0).add_(sq[:, None]).add_(sq[None, :])         # (one N x N float64 buffer: 3.2 GB at N = 20,000)
        d.clamp_(min=0.0).sqrt_()
        d.fill_diagonal_(0.0)
        return d                                                 # symmetric up to GEMM rounding (~1e-16)
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    sq = (Z * Z).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    np.maximum(d2, 0.0, out=d2)
    d = np.sqrt(d2)
    np.fill_diagonal(d, 0.0)
    return 0.5 * (d + d.T)


# ----------------------------------------------------------------------------- statistic
def energy_from_matrix(Dm, P, n1, n2):
    """E = 2 mean d(X,Y) - mean d(X,X) - mean d(Y,Y) (V form) for every column of P.
    Dm: (N,N) symmetric distance matrix (numpy or torch). P: (N,B) 0/1 matrix, column b = indicator of
    membership in sample 1 under labelling b (every column sums to n1)."""
    is_torch = not isinstance(Dm, np.ndarray)
    if is_torch:
        import torch
        P = torch.as_tensor(P, dtype=Dm.dtype, device=Dm.device)
    else:
        P = np.asarray(P, dtype=np.float64)
    S_all = Dm.sum()                          # 1' D 1
    DP = Dm @ P                               # (N,B)
    S1all = DP.sum(0)                         # 1' D 1_X  (D symmetric)
    S11 = (P * DP).sum(0)                     # 1_X' D 1_X   = sum of all n1^2 within-X distances
    S22 = S_all - 2.0 * S1all + S11           # 1_Y' D 1_Y   with 1_Y = 1 - 1_X
    S12 = S1all - S11                         # 1_X' D 1_Y   = sum of all n1 n2 cross distances
    E = 2.0 * S12 / (n1 * n2) - S11 / (n1 * n1) - S22 / (n2 * n2)
    return E.cpu().numpy() if is_torch else E


def energy_perm_test(X, Y, n_perm, rng, backend="numpy", return_perms=False):
    """Permutation test. rng: np.random.RandomState (or Generator) whose stream supplies the
    permutations in order, exactly as the original loop does. Returns (E_obs, p, E_perm)."""
    n1, n2 = len(X), len(Y)
    N = n1 + n2
    Dm = pairwise_dist(np.vstack([X, Y]), backend)
    perms = np.stack([rng.permutation(N) for _ in range(n_perm)])          # (B,N), same draws as original
    P = np.zeros((N, n_perm + 1), dtype=np.float64)
    P[:n1, 0] = 1.0                                                           # observed labelling
    P[perms[:, :n1].T, np.arange(1, n_perm + 1)[None, :]] = 1.0               # first n1 of each perm -> sample 1
    E = energy_from_matrix(Dm, P, n1, n2)
    E_obs, E_perm = float(E[0]), E[1:]
    check_statistics(E_obs, E_perm, "energy distance")     # release fix: never compare against a NaN observation
    p = float((np.sum(E_perm >= E_obs) + 1) / (n_perm + 1))
    return (E_obs, p, E_perm, perms) if return_perms else (E_obs, p, E_perm)


def energy_test_fast(emb_tr, emb_te, n_sub=1500, n_perm=500, seed=42, backend="numpy"):
    """Drop-in replacement for methods_v2.energy_distance_test: same signature, same draws, same output
    (obs, p)."""
    rng = np.random.RandomState(seed)
    X = emb_tr[rng.permutation(len(emb_tr))[:n_sub]]
    Y = emb_te[rng.permutation(len(emb_te))[:n_sub]]
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):  # release fix
        raise ValueError("energy_test_fast: the subsample contains NaN or inf")
    E_obs, p, _ = energy_perm_test(X, Y, n_perm, rng, backend)
    return E_obs, p


# ----------------------------------------------------------------------------- reference implementations
def energy_distance_test_original(emb_tr, emb_te, n_sub=1500, n_perm=500, seed=42):
    """VERBATIM copy of methods_v2.energy_distance_test (the paper's baseline), for the self-test only."""
    rng = np.random.RandomState(seed)
    X = emb_tr[rng.permutation(len(emb_tr))[:n_sub]]
    Y = emb_te[rng.permutation(len(emb_te))[:n_sub]]
    def ed_stat(X, Y):
        n, m = len(X), len(Y)
        xy = 0.0
        for s in range(0, n, 400):
            e = min(s+400, n)
            xy += np.sqrt(((X[s:e,None,:]-Y[None,:,:])**2).sum(2)).sum()
        xy /= (n*m)
        xx = 0.0
        for s in range(0, n, 400):
            e = min(s+400, n)
            xx += np.sqrt(((X[s:e,None,:]-X[None,:,:])**2).sum(2)).sum()
        xx /= (n*n)
        yy = 0.0
        for s in range(0, m, 400):
            e = min(s+400, m)
            yy += np.sqrt(((Y[s:e,None,:]-Y[None,:,:])**2).sum(2)).sum()
        yy /= (m*m)
        return float(2*xy - xx - yy)
    obs = ed_stat(X, Y)
    pooled = np.concatenate([X, Y]); n = len(X)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled))
        if ed_stat(pooled[p[:n]], pooled[p[n:]]) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def energy_naive(X, Y):
    from scipy.spatial.distance import cdist
    return 2 * cdist(X, Y).mean() - cdist(X, X).mean() - cdist(Y, Y).mean()


def selftest():
    ok = True
    print("A. fast vs verbatim original function (same seed => same subsample and permutations)")
    for trial, (n_tr, n_te, D, shift, n_sub, n_perm) in enumerate([(300, 200, 16, 0.0, 60, 100), (250, 250, 64, 0.15, 80, 150),
                                                                     (400, 300, 2048, 0.03, 50, 100)]):
        rng = np.random.default_rng(trial)
        tr = rng.standard_normal((n_tr, D)); te = rng.standard_normal((n_te, D)) + shift
        tr /= np.linalg.norm(tr, axis=1, keepdims=True); te /= np.linalg.norm(te, axis=1, keepdims=True)
        for dtype in (np.float64, np.float32):
            a, b = tr.astype(dtype), te.astype(dtype)
            for seed in (42, 43):
                o_obs, o_p = energy_distance_test_original(a, b, n_sub, n_perm, seed)
                for backend in ("numpy", "torch"):
                    try:
                        f_obs, f_p = energy_test_fast(a, b, n_sub, n_perm, seed, backend)
                    except ImportError:
                        continue
                    err = abs(f_obs - o_obs) / abs(o_obs)
                    good = (f_p == o_p) and err < (1e-9 if dtype is np.float64 else 1e-5)
                    ok &= good
                    print(f"  n_tr={n_tr} n_te={n_te} D={D:4d} shift={shift} n_sub={n_sub} B={n_perm} seed={seed} "
                          f"{np.dtype(dtype).name} {backend:6s}  obs {f_obs:+.6e} vs {o_obs:+.6e} (rel {err:.1e})  "
                          f"p {f_p:.4f} vs {o_p:.4f}  {'OK' if good else 'MISMATCH'}")
    print("B. fast vs brute-force (scipy cdist) with the same permutations, unequal sizes, duplicates")
    for trial, (n1, n2, D, shift) in enumerate([(60, 60, 8, 0.0), (50, 80, 32, 0.0), (70, 30, 16, 0.5)]):
        rng = np.random.default_rng(10 + trial)
        X = rng.standard_normal((n1, D)); Y = rng.standard_normal((n2, D)) + shift
        X[3] = X[7]; Y[1] = X[2]
        E_obs, p, E_perm, perms = energy_perm_test(X, Y, 200, np.random.RandomState(1), "numpy", return_perms=True)
        Z = np.vstack([X, Y])
        E_ref = energy_naive(X, Y); E_perm_ref = np.array([energy_naive(Z[pm[:n1]], Z[pm[n1:]]) for pm in perms])
        p_ref = (np.sum(E_perm_ref >= E_obs) + 1) / (len(perms) + 1)
        scale = max(abs(E_ref), np.max(np.abs(E_perm_ref)))
        err = max(abs(E_obs - E_ref), np.max(np.abs(E_perm - E_perm_ref))) / scale
        good = err < 1e-9 and p == p_ref
        ok &= good
        print(f"  n1={n1} n2={n2} D={D:3d} shift={shift}  max err/scale {err:.1e}  p {p:.4f} vs {p_ref:.4f}  {'OK' if good else 'MISMATCH'}")
    try:
        import dcor
        rng = np.random.default_rng(7); X = rng.standard_normal((30, 5)); Y = rng.standard_normal((25, 5)) + 0.2
        e_dcor = float(dcor.energy_distance(X, Y)); e_ours = energy_naive(X, Y)
        print(f"C. dcor.energy_distance {e_dcor:.10f} vs ours {e_ours:.10f} (rel {abs(e_dcor - e_ours) / e_dcor:.1e})")
        ok &= abs(e_dcor - e_ours) / e_dcor < 1e-9
    except ImportError:
        pass
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
    ap.add_argument("--n_perm", type=int, default=500)
    ap.add_argument("--seeds", default="42-51")
    ap.add_argument("--epochs", default="0-1000")
    ap.add_argument("--raw", action="store_true", help="skip L2 normalization")
    ap.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not (args.tag and args.npz_fmt and args.out):
        ap.error("--tag, --npz_fmt and --out are required (or use --selftest)")

    seeds = parse_range(args.seeds); epochs = parse_range(args.epochs)
    # release fix: resume by (epoch, seed), counting only valid rows of this run (tag, sample budget, permutation count,
    # feature normalization); rows of another run make the resume refuse. See mmd_tst_fast.py for the n_samples note.
    features = "raw" if args.raw else "l2"
    HEADER = ["tag", "n_samples", "n_perm", "epoch", "seed", "ed_val", "p", "features"]
    index = ResumeIndex(args.out, key_cols=("epoch", "seed"), value_cols=("ed_val", "p"), ranges={"p": (0, 1)},
                        expect={"tag": args.tag, "n_samples": args.n_samples, "n_perm": args.n_perm, "features": features},
                        header=HEADER)
    print(index.report())
    fout, w = open_append(args.out, HEADER)

    print(f"[{args.tag}] Energy Distance, n_sub={args.n_samples}/split, perms={args.n_perm}, seeds={seeds[0]}-{seeds[-1]}, "
          f"{'raw' if args.raw else 'L2-normalized'} features, backend={args.backend}", flush=True)
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
            ed_val, p = energy_test_fast(e_tr, e_te, args.n_samples, args.n_perm, seed, args.backend)
            w.writerow([args.tag, args.n_samples, args.n_perm, ep, seed, f"{ed_val:.8e}", f"{p:.6f}", features])
            ps.append(p)
        fout.flush()
        print(f"  ep{ep:4d}  med_p={np.median(ps):.4f}  reject {sum(p < 0.05 for p in ps)}/{len(ps)}  "
              f"first seed p={ps[0]:.4f}  ({time.time() - t0:.1f} s for {len(todo)} seeds)", flush=True)
    fout.close(); print("done")


if __name__ == "__main__":
    main()
