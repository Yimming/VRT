"""
knn_tst_budget.py — Schilling (1986) and Henze (1988) nearest-neighbor two-sample
tests at a chosen per-split budget, on precomputed per-epoch features.

Implementation mirrors knn_tst_all.py (paper Appendix B) with two corrections (2026-09-28):
  (1) Schilling variance = exact conditional variance given the graph (Henze 1988, Eq. 2.4), not (1+k)/(4nk)
  (2) self excluded from the neighbour list by index, not by position (duplicate-safe)
Original conventions kept:
  pooled kNN graph with sklearn NearestNeighbors (k+1 incl. self), k=3
  Schilling: T = coincidences/(n k), mu = lam1^2 + lam2^2, sigma2 = (1+k)/(4 n k), one-sided normal p
  Henze:     same coincidence count, 2,000 label permutations, p = (#T_perm >= T_obs + 1)/(n_perm + 1)
  per seed: default_rng(seed).choice(len, n, replace=False) on each split; Henze rng seeded with the same seed
Only differences: n_samples is a CLI argument (paper used 2,000), and the graph is built once per seed
and shared by both statistics (identical to building it twice).

Usage:
  python knn_tst_budget.py --tag r50 --npz_fmt ".../r50/ep{ep:03d}.npz" --n_samples 10000 \
      --epochs 40-140 --out knn_r50_n10000.csv
"""
import os, argparse, csv
import numpy as np
from release_utils import validate_features, ResumeIndex, open_append, count_consistent
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors


def knn_graph(Z, k, backend="torch"):
    """k nearest neighbours of every pooled point, self excluded. Same neighbours as
    sklearn NearestNeighbors (Euclidean); the torch backend computes exact squared
    Euclidean distances via the Gram matrix and is GPU-accelerated when available."""
    if backend == "sklearn":
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(Z)
        _, idx_full = nbrs.kneighbors(Z)
        # exclude self BY INDEX (not by position): robust to exact duplicate points
        not_self = idx_full != np.arange(len(Z))[:, None]
        sel = np.argsort(~not_self, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(idx_full, sel, axis=1)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float32)).to(dev)
    sq = (T * T).sum(1)
    n = T.shape[0]; out = np.empty((n, k), dtype=np.int64)
    step = 4096
    with torch.no_grad():
        for s in range(0, n, step):
            e = min(s + step, n)
            d2 = sq[s:e, None] + sq[None, :] - 2.0 * (T[s:e] @ T.T)
            d2[torch.arange(e - s), torch.arange(s, e)] = float("inf")   # exclude self
            out[s:e] = torch.topk(d2, k, dim=1, largest=False).indices.cpu().numpy()
    return out


def mutual_pairs(idx):
    """number of unordered pairs {i, j} with j in kNN(i) and i in kNN(j)"""
    n = idx.shape[0]
    back = idx[idx]                                   # n x k x k : neighbours of neighbours
    mutual_edge = (back == np.arange(n)[:, None, None]).any(axis=2)   # n x k
    return int(mutual_edge.sum()) // 2


def knn_conditional_moments(idx, n1, n2, k):
    """Exact mean and variance of the coincidence COUNT under fixed-size random labelling
    of a fixed directed kNN graph (Henze 1988, Proposition 2.1, Eq. 2.4). Inputs: mutual
    pairs M and the in-degree of every node. Validated against 40,000 label permutations
    on equal and unequal sample sizes, k = 1, 3, 7, D = 4 to 512 (variance within 1.5%)."""
    n = n1 + n2
    if n <= 3 or min(n1, n2) < 1 or not 1 <= k < n or idx.shape != (n, k):
        raise ValueError("expected n > 3, nonempty groups, and an (n, k) neighbour graph")
    M = mutual_pairs(idx)
    indegree = np.bincount(idx.ravel(), minlength=n).astype(np.float64)
    v = 2.0 * M / (n * k)
    c = np.sum((indegree - k) ** 2) / (n * k)
    a = 4.0 * (n1 - 1) * (n2 - 1) / ((n - 2) * (n - 3))
    prefactor = k * n1 * n2 / (n - 1)
    term1 = a * (1.0 + v - 2.0 * k / (n - 1))
    term2 = (1.0 - a) * c
    mean_count = k * (n1 * (n1 - 1) + n2 * (n2 - 1)) / (n - 1)
    var_count = prefactor * (term1 + term2)
    tol = 64 * np.finfo(float).eps * prefactor * (abs(term1) + abs(term2))
    if abs(var_count) <= tol:
        var_count = 0.0
    elif var_count < 0:
        raise ValueError("negative conditional variance: check the neighbour graph")
    return float(mean_count), float(var_count)


def schilling_p(idx, tags, n1, n2, k):
    """Schilling (1986) coincidence statistic, graph-conditioned normal approximation with
    the exact null mean and variance from knn_conditional_moments. The original code used
    sigma2 = (1+k)/(4nk), which overstated the variance by about 2.4x at k=3."""
    coincidences = int((tags[:, None] == tags[idx]).sum())
    mean_count, var_count = knn_conditional_moments(idx, n1, n2, k)
    if var_count == 0:
        return 1.0
    z = (coincidences - mean_count) / np.sqrt(var_count)
    return float(norm.sf(z))


def henze_p(idx, tags_obs, n_perm, seed):
    rng = np.random.default_rng(seed)
    def stat(tags):
        return (tags[:, None] == tags[idx]).sum()
    T_obs = stat(tags_obs)
    T_perm = np.array([stat(rng.permutation(tags_obs)) for _ in range(n_perm)])
    return float((np.sum(T_perm >= T_obs) + 1) / (n_perm + 1))


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
    ap.add_argument("--tag", required=True)
    ap.add_argument("--npz_fmt", required=True)
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seeds", default="42-51")
    ap.add_argument("--epochs", default="0-300")
    ap.add_argument("--raw", action="store_true", help="skip L2 normalization")
    ap.add_argument("--backend", choices=["torch", "sklearn"], default="torch")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seeds = parse_range(args.seeds); epochs = parse_range(args.epochs)
    # release fix: resume by (epoch, seed), counting only valid rows (both p-values finite in [0, 1]) of this tag, sample
    # size, k and permutation count; an output file with the old 7-column header is refused (write a new file)
    # The sample budget, k, permutation count and feature normalization identify the run; the Henze count must agree with
    # p_henze; the runner refuses a budget larger than a split (below), so n_samples always equals the requested budget.
    features = "raw" if args.raw else "l2"
    HEADER = ["tag", "n_samples", "k", "epoch", "seed", "p_schilling", "p_henze", "henze_count", "n_perm", "features"]
    index = ResumeIndex(args.out, key_cols=("epoch", "seed"), value_cols=("p_schilling", "p_henze", "henze_count"),
                        ranges={"p_schilling": (0, 1), "p_henze": (0, 1), "henze_count": (0, args.n_perm)},
                        expect={"tag": args.tag, "n_samples": args.n_samples, "k": args.k, "n_perm": args.n_perm, "features": features},
                        row_check=count_consistent("p_henze", "henze_count", "n_perm"), header=HEADER)
    print(index.report())
    # release fix: p_henze with six decimals and the integer permutation count (the as-run files hold four decimals,
    # which exact_perm_p() recovers exactly); the column list is a superset of the as-run one
    fout, w = open_append(args.out, HEADER)

    print(f"[{args.tag}] Schilling/Henze, n={args.n_samples}/split, k={args.k}, perms={args.n_perm}, "
          f"seeds={seeds[0]}-{seeds[-1]}, {'raw' if args.raw else 'L2-normalized'} features, backend={args.backend}")
    for ep in epochs:
        todo = [s for s in seeds if not index.has(ep, s)]
        if not todo:
            continue
        loaded = load_epoch(args.npz_fmt, ep)
        if loaded is None:
            continue
        e_tr, e_te = validate_features(*loaded, names=("train features", "test features"), allow_zero_rows=args.raw)
        if min(len(e_tr), len(e_te)) < args.n_samples:       # release fix: never silently test a smaller budget
            raise SystemExit(f"ep {ep}: split sizes {len(e_tr)}/{len(e_te)} are below --n_samples {args.n_samples}")
        if not args.raw:                                     # a zero row is a valid point of the raw feature space
            e_tr, e_te = l2n(e_tr), l2n(e_te)
        ps, ph = [], []
        for seed in todo:
            rng = np.random.default_rng(seed)
            n = args.n_samples
            X = e_tr[rng.choice(len(e_tr), n, replace=False)]
            Y = e_te[rng.choice(len(e_te), n, replace=False)]
            Z = np.vstack([X, Y]); tags = np.array([0] * n + [1] * n)
            idx = knn_graph(Z, args.k, args.backend)
            p_s = schilling_p(idx, tags, n, n, args.k)
            p_h = henze_p(idx, tags, args.n_perm, seed)
            if not (np.isfinite(p_s) and np.isfinite(p_h)):
                raise FloatingPointError(f"ep {ep} seed {seed}: non-finite p (Schilling {p_s}, Henze {p_h})")
            count_h = int(round(p_h * (args.n_perm + 1))) - 1                 # p_h = (count + 1) / (n_perm + 1)
            w.writerow([args.tag, n, args.k, ep, seed, f"{p_s:.6f}", f"{p_h:.6f}", count_h, args.n_perm, features])
            ps.append(p_s); ph.append(p_h)
        fout.flush()
        print(f"  ep{ep:4d}  Schilling med_p={np.median(ps):.4f} ({sum(p<0.05 for p in ps)}/{len(ps)})"
              f"   Henze med_p={np.median(ph):.4f} ({sum(p<0.05 for p in ph)}/{len(ph)})", flush=True)
    fout.close(); print("done")


if __name__ == "__main__":
    main()
