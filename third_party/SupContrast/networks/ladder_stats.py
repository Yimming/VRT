"""
ladder_stats.py — statistic ladder + fair-budget table (Stage 2, CPU only)
==========================================================================
Consumes the raw-feature npz files produced by extract_feats.py and runs,
under an IDENTICAL protocol (same reference/query splits, same median-
difference statistic, same permutation test), a ladder of per-point
statistics from simplest to full VRT:

  norm       : s(q) = ||q||            (raw norm; location test)
  norm_disp  : s(q) = ||q||            (dispersion test: MAD difference)
  cos_topk   : s(q) = mean cosine to k nearest reference neighbors
               (reviewer 9fxN's proposed simple baseline)
  theta_k    : s(q) = angle to the k-th nearest reference neighbor
  L_k        : s(q) = log spherical-cap volume at the k-th neighbor
  chat       : s(q) = VRT score (Renyi log-volume spacings)  [full method]

plus component-isolation / fair-budget baselines:

  mmd_raw    : MMD on raw l2-normalized embeddings, n_sub per split
  mmd_chat   : MMD on the scalar chat scores, n_sub per split
  meandiff   : mean (not median) difference on chat

Budgets:
  paper : M=2000 reference + 8000 train queries, all 10000 test queries
          (total train access 10k — matches the paper protocol)
  low   : M=500 + 1000 train queries, 1500 test queries
          (total-access matched to the 1500-per-split baseline budget)

Fair-budget MMD: --mmd_sub controls the per-split subsample of mmd_raw
(1500 = paper's baseline budget; 10000 = equal-budget). MMD uses a cached
pooled kernel + permutation by index sums — mathematically identical to
the supplementary implementation, orders of magnitude faster.

Pure NumPy/SciPy. Runs on a laptop for v7/ablA; use a CPU node for r50.

Usage:
  python ladder_stats.py --model v7   --feats_dir <...>/ladder_feats/v7
  python ladder_stats.py --model r50  --feats_dir <...>/ladder_feats/r50 \
      --epochs 0-200 --mmd_full_every 5
Output: CSV  <out>/ladder_{model}.csv  with columns
  model,epoch,budget,stat,seed,T,p
"""
import argparse
import os
import re
import time

import numpy as np
from scipy.special import betainc, betaln

SEED0 = 42          # paper convention: seeds 42..51 (baseline_multiseed.py)
K = 60
N_PERM = 2000


# ── VRT scoring pieces (identical math to supplementary methods.py) ───
def log_cap_volume(theta, D):
    sin2 = np.sin(theta)**2
    I = betainc((D-1)/2.0, 0.5, sin2)
    V = np.where(theta <= np.pi/2, 0.5*I, 1.0 - 0.5*I)
    safe = V > 1e-300
    log_V = np.empty_like(V)
    log_V[safe] = np.log(V[safe])
    if np.any(~safe):
        a = (D-1)/2.0
        const = -np.log(2.0) - np.log(a) - betaln(a, 0.5)
        log_V[~safe] = const + (D-1)*np.log(np.maximum(np.sin(theta[~safe]), 1e-300))
    return log_V


def topk_cosines(query, ref, k, batch=2048):
    """Sorted (desc) top-k cosine similarities of each query against ref."""
    out = []
    for s in range(0, query.shape[0], batch):
        cos = query[s:s+batch] @ ref.T
        pidx = np.argpartition(-cos, k-1, axis=1)[:, :k]
        top = np.take_along_axis(cos, pidx, axis=1)
        top = np.take_along_axis(top, np.argsort(-top, axis=1), axis=1)
        out.append(top)
    return np.concatenate(out, 0)


def ladder_scores(top, D, k):
    """From a top-k cosine matrix, derive all angular ladder statistics."""
    theta = np.arccos(np.clip(top, -1, 1))          # (n, k) ascending in angle
    L = log_cap_volume(theta, D)                     # (n, k)
    dL = np.diff(L, axis=1)
    r = np.arange(1, k, dtype=np.float64)
    Z = np.maximum(r[None, :] * dL, 0.0)
    return {
        'cos_topk': top.mean(axis=1),                # mean cosine to top-k
        'theta_k': theta[:, -1],                     # k-th neighbor angle
        'L_k': L[:, -1],                             # cap volume at k-th
        'chat': np.maximum(Z.mean(axis=1), 1e-12),   # full VRT score
    }


# ── generic permutation tests on precomputed per-point scores ─────────
def perm_test(s_tr, s_te, stat_fn, n_perm=N_PERM, seed=0):
    rng = np.random.RandomState(seed)
    obs = stat_fn(s_tr, s_te)
    pooled = np.concatenate([s_tr, s_te])
    na = len(s_tr)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled))
        if abs(stat_fn(pooled[p[:na]], pooled[p[na:]])) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def med_diff(a, b):
    return np.median(b) - np.median(a)


def mean_diff(a, b):
    return np.mean(b) - np.mean(a)


def mad_diff(a, b):
    mad = lambda x: np.median(np.abs(x - np.median(x)))
    return mad(b) - mad(a)


# ── MMD with cached kernel (identical math, fast permutations) ────────
def mmd_cached(X, Y, n_perm=1000, seed=42):
    """Gaussian-RBF MMD^2 (unbiased), median heuristic on <=2000 pooled
    points, permutation p-value. Same math as supplementary mmd_test but
    the pooled kernel is computed once."""
    rng = np.random.RandomState(seed)
    pooled = np.concatenate([X, Y]).astype(np.float64)
    n, m = len(X), len(Y)
    sub = pooled[rng.choice(len(pooled), min(2000, len(pooled)), replace=False)]
    d2s = np.sum(sub**2, 1, keepdims=True) + np.sum(sub**2, 1, keepdims=True).T - 2*sub@sub.T
    d2s = np.maximum(d2s, 0.0)
    np.fill_diagonal(d2s, np.inf)
    vals = d2s[np.isfinite(d2s) & (d2s > 0)]
    sigma = max(float(np.sqrt(np.median(vals))), 1e-8) if len(vals) else 1.0

    sq = np.sum(pooled**2, 1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2*pooled@pooled.T, 0.0)
    Km = np.exp(-d2 / (2*sigma**2)).astype(np.float32)
    np.fill_diagonal(Km, 0.0)
    tot = Km.sum()
    rows = Km.sum(axis=1)

    def mmd2_from_mask(zx):
        # zx: boolean mask of "X" membership, size n+m with n True
        sx = rows[zx].sum()
        Kxx = Km[np.ix_(zx, zx)].sum()
        Kyy = tot - 2*sx + Kxx
        Kxy = sx - Kxx
        return Kxx/(n*(n-1)) + Kyy/(m*(m-1)) - 2*Kxy/(n*m)

    base = np.zeros(n+m, dtype=bool)
    base[:n] = True
    obs = mmd2_from_mask(base)
    count = 0
    for _ in range(n_perm):
        z = np.zeros(n+m, dtype=bool)
        z[rng.permutation(n+m)[:n]] = True
        if mmd2_from_mask(z) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


# ── budgets ───────────────────────────────────────────────────────────
BUDGETS = {
    'paper': dict(M=2000, n_qtr=8000, n_qte=None),   # None = all test
    'low':   dict(M=500,  n_qtr=1000, n_qte=1500),
}


def parse_epochs(spec, available):
    if spec is None:
        return available
    keep = set()
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            keep.update(range(int(a), int(b)+1))
        else:
            keep.add(int(part))
    return [e for e in available if e in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--feats_dir', required=True)
    ap.add_argument('--out_dir', default='.')
    ap.add_argument('--epochs', default=None, help='e.g. "0-60,100,150-200"')
    ap.add_argument('--mmd_sub', type=int, default=1500,
                    help='per-split subsample for mmd_raw (paper baseline '
                         'budget); use 10000 for the equal-budget run')
    ap.add_argument('--mmd_full_every', type=int, default=1,
                    help='run mmd_raw only every N epochs (cost control '
                         'when mmd_sub is large)')
    ap.add_argument('--seeds', type=int, default=10)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.feats_dir)
                   if re.fullmatch(r'ep\d+\.npz', f))
    eps = [int(f[2:-4]) for f in files]
    eps = parse_epochs(args.epochs, eps)
    print(f'{args.model}: {len(eps)} epochs to process')

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f'ladder_{args.model}.csv')
    new = not os.path.exists(out_csv)
    fout = open(out_csv, 'a')
    if new:
        fout.write('model,epoch,budget,stat,seed,T,p\n')

    def emit(ep, budget, stat, seed, T, p):
        fout.write(f'{args.model},{ep},{budget},{stat},{seed},{T:+.6e},{p:.6f}\n')
        fout.flush()

    for ep in eps:
        t0 = time.time()
        z = np.load(os.path.join(args.feats_dir, f'ep{ep:03d}.npz'))
        h_tr, h_te = z['h_tr'].astype(np.float64), z['h_te'].astype(np.float64)
        D = h_tr.shape[1]
        norms_tr = np.linalg.norm(h_tr, axis=1)
        norms_te = np.linalg.norm(h_te, axis=1)
        e_tr, e_te = l2n(h_tr), l2n(h_te)

        for bname, B in BUDGETS.items():
            for seed in range(SEED0, SEED0 + args.seeds):
                rng = np.random.RandomState(seed)
                perm = rng.permutation(len(e_tr))
                ref_idx = perm[:B['M']]
                qtr_idx = perm[B['M']:B['M'] + B['n_qtr']]
                if B['n_qte'] is None:
                    qte_idx = np.arange(len(e_te))
                else:
                    qte_idx = rng.permutation(len(e_te))[:B['n_qte']]

                ref = e_tr[ref_idx]
                # norm rungs use ref+query train samples so the train-side
                # ACCESS is identical to every other statistic in this tier
                # (paper: 2000+8000=10000 vs 10000; low: 500+1000=1500 vs 1500)
                ntr = norms_tr[np.concatenate([ref_idx, qtr_idx])]
                nte = norms_te[qte_idx]
                T, p = perm_test(ntr, nte, med_diff, seed=seed)
                emit(ep, bname, 'norm', seed, T, p)
                T, p = perm_test(ntr, nte, mad_diff, seed=seed)
                emit(ep, bname, 'norm_disp', seed, T, p)

                # one kNN pass serves all four angular rungs
                k = min(K, B['M'])
                top_tr = topk_cosines(e_tr[qtr_idx], ref, k)
                top_te = topk_cosines(e_te[qte_idx], ref, k)
                s_tr = ladder_scores(top_tr, D, k)
                s_te = ladder_scores(top_te, D, k)
                for stat in ('cos_topk', 'theta_k', 'L_k', 'chat'):
                    T, p = perm_test(s_tr[stat], s_te[stat], med_diff, seed=seed)
                    emit(ep, bname, stat, seed, T, p)

                # chat variants (paper budget only, component isolation)
                if bname == 'paper':
                    T, p = perm_test(s_tr['chat'], s_te['chat'], mean_diff, seed=seed)
                    emit(ep, bname, 'meandiff_chat', seed, T, p)
                    if seed < SEED0 + 5:
                        c_tr, c_te = s_tr['chat'], s_te['chat']
                        sub = min(args.mmd_sub, len(c_tr), len(c_te))
                        _, p = mmd_cached(c_tr[rng.permutation(len(c_tr))[:sub], None],
                                          c_te[rng.permutation(len(c_te))[:sub], None],
                                          seed=seed)
                        emit(ep, bname, 'mmd_chat', seed, np.nan, p)

            # MMD on raw normalized embeddings (fair-budget baseline);
            # budget-independent protocol, so run once per epoch under 'paper'
            if bname == 'paper' and ep % args.mmd_full_every == 0:
                for seed in range(SEED0, SEED0 + 5):
                    rng = np.random.RandomState(seed)
                    nx = min(args.mmd_sub, len(e_tr))
                    ny = min(args.mmd_sub, len(e_te))
                    X = e_tr[rng.permutation(len(e_tr))[:nx]]
                    Y = e_te[rng.permutation(len(e_te))[:ny]]
                    _, p = mmd_cached(X, Y, seed=seed)
                    emit(ep, bname, f'mmd_raw{args.mmd_sub}', seed, np.nan, p)

        print(f'ep{ep:03d} done in {time.time()-t0:.1f}s', flush=True)

    fout.close()
    print('->', out_csv)


if __name__ == '__main__':
    main()
