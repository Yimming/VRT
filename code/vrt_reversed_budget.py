"""
vrt_reversed_budget.py — VRT with the TEST split as reference, at the two
matched-budget regimes used in the rebuttal (Ftg8 W1), on precomputed features.

Forward VRT (paper): reference drawn from TRAIN, queries = rest of train + test.
Reversed VRT (this): reference drawn from TEST,  queries = rest of test + train.
Statistic, calibration, permutation procedure and onset rule are unchanged.

Budgets (mirror of the forward protocol):
  large : 10,000 per split. ref M=2000 from test, q_te = 8000 remaining test,
          q_tr = 8000 subsampled train, k=60
  small : 1,500 per split.  ref M=500 from a 1,500 test subsample, q_te = 1,000
          remaining, q_tr = 1,500 train, k=15  (k scales with M, 60/2000 = 15/500)

Features: one .npy per epoch and split, raw or normalized (script L2-normalizes).
Point the script at them with python format strings containing {ep}, e.g.
  --tr_fmt "$FEATS/r50/ep{ep:03d}_train.npy" --te_fmt "$FEATS/r50/ep{ep:03d}_test.npy"

Usage:
  python vrt_reversed_budget.py --tag r50 --budget large \
      --tr_fmt ".../ep{ep:03d}_train.npy" --te_fmt ".../ep{ep:03d}_test.npy" \
      --epochs 0-200 --out vrt_rev_r50_large.csv
Resumes: epochs already present in --out with >= n_seeds rows are skipped.
"""
import os, argparse, csv, glob
import numpy as np
from release_utils import validate_features, ResumeIndex, open_append, check_scores, check_perm_statistic, count_consistent
from scipy.special import betainc, betaln

# ── VRT core (identical to the released src/methods.py) ─────────────────
def log_cap_volume(theta, D):
    sin2 = np.sin(theta) ** 2
    I = betainc((D - 1) / 2.0, 0.5, sin2)
    V = np.where(theta <= np.pi / 2, 0.5 * I, 1.0 - 0.5 * I)
    safe = V > 1e-300
    log_V = np.empty_like(V)
    log_V[safe] = np.log(V[safe])
    if np.any(~safe):
        a = (D - 1) / 2.0
        const = -np.log(2.0) - np.log(a) - betaln(a, 0.5)
        log_V[~safe] = const + (D - 1) * np.log(np.maximum(np.sin(theta[~safe]), 1e-300))
    return log_V


def compute_chat(query, ref, D, k, batch_size=1024):
    if k > ref.shape[0]:
        raise ValueError(f"k={k} exceeds reference size M={ref.shape[0]}")
    Nq = query.shape[0]; theta_all = []
    for s in range(0, Nq, batch_size):
        e = min(s + batch_size, Nq)
        cos = query[s:e] @ ref.T
        pidx = np.argpartition(-cos, k - 1, axis=1)[:, :k]
        top = np.take_along_axis(cos, pidx, axis=1)
        top = np.take_along_axis(top, np.argsort(-top, axis=1), axis=1)
        theta_all.append(np.arccos(np.clip(top, -1, 1)))
    theta = np.concatenate(theta_all, 0)
    L = log_cap_volume(theta, D); dL = np.diff(L, axis=1)
    r = np.arange(1, k, dtype=np.float64)
    Z = np.maximum(r[None, :] * dL, 0.0)
    return np.maximum(Z.mean(axis=1), 1e-12)


def perm_p(c_tr, c_te, n_perm, rng):
    """release fix: scores and median differences are checked for NaN/inf; draws and arithmetic unchanged"""
    check_scores(c_tr, c_te)
    obs = check_perm_statistic(np.median(c_te) - np.median(c_tr), "observed median difference")
    pooled = np.concatenate([c_tr, c_te]); na = len(c_tr)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled))
        d = check_perm_statistic(np.median(pooled[p[na:]]) - np.median(pooled[p[:na]]), "permuted median difference")
        if abs(d) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def vrt_reversed(emb_tr, emb_te, D, M, k, n_q, n_perm, seed):
    """Reference from TEST. n_q = query count per side (train side subsampled to n_q,
    test side = remaining after the reference, capped at n_q)."""
    rng = np.random.RandomState(seed)
    pte = rng.permutation(len(emb_te))
    ref = emb_te[pte[:M]]
    q_te = emb_te[pte[M:M + n_q]]
    ptr = rng.permutation(len(emb_tr))
    q_tr = emb_tr[ptr[:n_q]]
    c_tr = compute_chat(q_tr, ref, D, k)
    c_te = compute_chat(q_te, ref, D, k)
    return perm_p(c_tr, c_te, n_perm, rng)


BUDGETS = {
    # name : (train_pool, test_pool, M_ref, k, n_query_per_side)
    "large": (10000, 10000, 2000, 60, 8000),
    "small": (1500,  1500,  500,  15, 1500),
}


def l2n(x):
    x = x.astype(np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


KEY_PAIRS = [("X_tr", "X_te"), ("tr", "te"), ("train", "test"), ("emb_tr", "emb_te"),
             ("Xtr", "Xte"), ("x_tr", "x_te"), ("feat_tr", "feat_te"), ("h_tr", "h_te")]


def load_epoch(args, ep):
    """Return (X_tr, X_te) or None if files are missing. Two layouts:
    (a) --npz_fmt : one .npz per epoch holding both splits (keys auto-detected or --tr_key/--te_key)
    (b) --tr_fmt/--te_fmt : one .npy per split per epoch"""
    if args.npz_fmt:
        f = args.npz_fmt.format(ep=ep)
        if not os.path.exists(f):
            return None
        d = np.load(f)
        if args.tr_key:
            return d[args.tr_key], d[args.te_key]
        for a, b in KEY_PAIRS:
            if a in d.files and b in d.files:
                return d[a], d[b]
        raise KeyError(f"cannot identify train/test arrays in {f}; keys = {d.files}. "
                       f"Pass --tr_key/--te_key explicitly.")
    ftr, fte = args.tr_fmt.format(ep=ep), args.te_fmt.format(ep=ep)
    if not (os.path.exists(ftr) and os.path.exists(fte)):
        return None
    return np.load(ftr), np.load(fte)


def parse_epochs(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--budget", choices=list(BUDGETS), required=True)
    ap.add_argument("--npz_fmt", default="", help="format string with {ep}: one .npz per epoch holding both splits")
    ap.add_argument("--tr_key", default="", help="npz key of train features (auto-detected if omitted)")
    ap.add_argument("--te_key", default="", help="npz key of test features")
    ap.add_argument("--tr_fmt", default="", help="alternative layout: format string with {ep}, train .npy")
    ap.add_argument("--te_fmt", default="", help="alternative layout: format string with {ep}, test .npy")
    ap.add_argument("--epochs", default="0-200")
    ap.add_argument("--seeds", default="42-51")
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.npz_fmt and not (args.tr_fmt and args.te_fmt):
        ap.error("give --npz_fmt, or both --tr_fmt and --te_fmt")

    tr_pool, te_pool, M, k, n_q = BUDGETS[args.budget]
    seeds = parse_epochs(args.seeds)
    epochs = parse_epochs(args.epochs)

    # release fix: resume by (epoch, seed), counting only valid rows of this tag and budget; p with six decimals and
    # the permutation count (the as-run files hold four decimals; exact_perm_p() recovers the exact value, see
    # release_utils and the Appendix G correction). An output file with the old 9-column header is refused.
    HEADER = ["tag", "budget", "direction", "epoch", "seed", "M", "k", "delta_c", "p", "count", "n_perm"]
    index = ResumeIndex(args.out, key_cols=("epoch", "seed"), value_cols=("delta_c", "p", "count"),
                        ranges={"p": (0, 1), "count": (0, args.n_perm)},
                        expect={"tag": args.tag, "budget": args.budget, "direction": "test_ref", "M": M, "k": k,
                                "n_perm": args.n_perm}, row_check=count_consistent("p", "count", "n_perm"), header=HEADER)
    print(index.report())
    fout, w = open_append(args.out, HEADER)

    print(f"[{args.tag}] reversed VRT, budget={args.budget} (pool {tr_pool}/{te_pool}, "
          f"M={M} from TEST, k={k}, q={n_q}/side), seeds={seeds[0]}-{seeds[-1]}, perms={args.n_perm}")
    for ep in epochs:
        todo = [s for s in seeds if not index.has(ep, s)]
        if not todo:
            continue
        loaded = load_epoch(args, ep)
        if loaded is None:
            continue
        validate_features(*loaded, names=("train features", "test features"))
        X_tr, X_te = l2n(loaded[0]), l2n(loaded[1])
        D = X_tr.shape[1]
        pvals = []
        for sd in todo:
            rng = np.random.RandomState(1000 + sd)          # budget subsample draw
            tr = X_tr[rng.permutation(len(X_tr))[:tr_pool]]
            te = X_te[rng.permutation(len(X_te))[:te_pool]]
            dc, p = vrt_reversed(tr, te, D, M, k, n_q, args.n_perm, sd)
            count = int(round(p * (args.n_perm + 1))) - 1
            w.writerow([args.tag, args.budget, "test_ref", ep, sd, M, k, f"{dc:.6g}", f"{p:.6f}", count, args.n_perm])
            pvals.append(p)
        fout.flush()
        med = float(np.median(pvals)); nsig = sum(p < 0.05 for p in pvals)
        print(f"  ep{ep:4d}  med_p={med:.4f}  {nsig:2d}/{len(seeds)} < 0.05", flush=True)
    fout.close()
    print("done")


if __name__ == "__main__":
    main()
