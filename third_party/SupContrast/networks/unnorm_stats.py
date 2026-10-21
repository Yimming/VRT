"""
unnorm_stats.py — baselines & simple statistics on UNNORMALIZED features
========================================================================
Answers nxR4 Q1 ("do the gains arise from intrinsic geometry or from the
normalization itself?") and Ftg8's sample-budget concern, using the same
raw npz features as ladder_stats.py.

For each checkpoint and each space in {unnorm, l2}:

  MMD (Gaussian RBF, median heuristic, permutation p):
    - n_sub = 1500 per split   (paper's baseline budget)
    - n_sub = 10000 per split  (EQUAL budget: matches VRT's total access
      of 10k train + 10k test; cached-kernel + matvec permutations,
      n_perm reduced to --nperm_big for tractability)

  Euclidean-kNN per-point statistics on UNNORM space only (native-space
  analogues of the ladder rungs; same protocol, same two budgets,
  10 seeds, median-difference + permutation):
    dist_topk : mean Euclidean distance to k nearest reference points
    r_k       : distance to the k-th nearest reference point
    logV_k    : D * log r_k   (Euclidean-ball log-volume at k-th neighbor)
    chat_eucl : mean of (j-1)*D*(log r_j - log r_{j-1}), j=2..k
                (Euclidean-ball analogue of the VRT score)

Budgets for per-point statistics (identical to ladder_stats.py):
  paper : M=2000 reference + 8000 train queries + all 10000 test queries
  low   : M=500 + 1000 train queries + 1500 test queries

Output CSV: unnorm_{model}.csv with
  model,epoch,space,budget,stat,seed,T,p

Usage:
  python unnorm_stats.py --model v7  --feats_dir <...>/ladder_feats/v7
  python unnorm_stats.py --model r50 --feats_dir <...>/ladder_feats/r50 \
      --mmd10k_every 5
"""
import argparse
import os
import re
import time

import numpy as np

SEED0 = 42          # paper convention: seeds 42..51
SEEDS_POINT = 10
N_PERM_POINT = 2000
K = 60

BUDGETS = {
    'paper': dict(M=2000, n_qtr=8000, n_qte=None),
    'low':   dict(M=500,  n_qtr=1000, n_qte=1500),
}


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


# ── per-point permutation test (median difference) ────────────────────
def perm_test_med(s_tr, s_te, n_perm=N_PERM_POINT, seed=0):
    rng = np.random.RandomState(seed)
    obs = np.median(s_te) - np.median(s_tr)
    pooled = np.concatenate([s_tr, s_te])
    na = len(s_tr)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled))
        if abs(np.median(pooled[p[na:]]) - np.median(pooled[p[:na]])) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)


# ── Euclidean top-k distances ─────────────────────────────────────────
def topk_eucl(query, ref, k, batch=2048):
    """Sorted (asc) top-k Euclidean distances of each query against ref."""
    rsq = np.sum(ref**2, axis=1)
    out = []
    for s in range(0, query.shape[0], batch):
        q = query[s:s+batch]
        d2 = np.maximum(np.sum(q**2, 1, keepdims=True) + rsq[None, :] - 2*q@ref.T, 0.0)
        pidx = np.argpartition(d2, k-1, axis=1)[:, :k]
        top = np.take_along_axis(d2, pidx, axis=1)
        top = np.take_along_axis(top, np.argsort(top, axis=1), axis=1)
        out.append(np.sqrt(top))
    return np.concatenate(out, 0)


def eucl_scores(dist, D):
    """Native-space ladder statistics from sorted kNN distances."""
    logr = np.log(np.maximum(dist, 1e-12))
    dlog = np.diff(logr, axis=1)
    j = np.arange(1, dist.shape[1], dtype=np.float64)
    Z = np.maximum(j[None, :] * D * dlog, 0.0)
    return {
        'dist_topk': dist.mean(axis=1),
        'r_k': dist[:, -1],
        'logV_k': D * logr[:, -1],
        'chat_eucl': np.maximum(Z.mean(axis=1), 1e-12),
    }


# ── MMD with cached kernel and matvec permutations ────────────────────
def mmd_fast(X, Y, n_perm, seed=42):
    rng = np.random.RandomState(seed)
    pooled = np.concatenate([X, Y]).astype(np.float64)
    n, m = len(X), len(Y)
    N = n + m
    sub = pooled[rng.choice(N, min(2000, N), replace=False)]
    d2s = np.sum(sub**2, 1, keepdims=True) + np.sum(sub**2, 1, keepdims=True).T - 2*sub@sub.T
    d2s = np.maximum(d2s, 0.0)
    np.fill_diagonal(d2s, np.inf)
    vals = d2s[np.isfinite(d2s) & (d2s > 0)]
    sigma = max(float(np.sqrt(np.median(vals))), 1e-8) if len(vals) else 1.0

    sq = np.sum(pooled**2, 1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2*pooled@pooled.T, 0.0)
    Km = np.exp(-d2 / (2*sigma**2)).astype(np.float32)
    del d2
    np.fill_diagonal(Km, 0.0)
    tot = float(Km.sum())

    def stat(z):
        # z: float32 0/1 vector with n ones. One matvec per permutation.
        S = Km @ z
        Kxx = float(z @ S)
        sx = float(S.sum())          # = z^T K 1
        Kyy = tot - 2*sx + Kxx
        Kxy = sx - Kxx
        return Kxx/(n*(n-1)) + Kyy/(m*(m-1)) - 2*Kxy/(n*m)

    z0 = np.zeros(N, dtype=np.float32)
    z0[:n] = 1.0
    obs = stat(z0)
    count = 0
    for _ in range(n_perm):
        z = np.zeros(N, dtype=np.float32)
        z[rng.permutation(N)[:n]] = 1.0
        if stat(z) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


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
    ap.add_argument('--epochs', default=None)
    ap.add_argument('--mmd_seeds', type=int, default=5)
    ap.add_argument('--nperm_small', type=int, default=1000)
    ap.add_argument('--nperm_big', type=int, default=200,
                    help='permutations for the 10k equal-budget MMD')
    ap.add_argument('--mmd10k_every', type=int, default=1,
                    help='run the equal-budget MMD only every N epochs')
    ap.add_argument('--skip_eucl', action='store_true')
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.feats_dir)
                   if re.fullmatch(r'ep\d+\.npz', f))
    eps = parse_epochs(args.epochs, [int(f[2:-4]) for f in files])
    print(f'{args.model}: {len(eps)} epochs')

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f'unnorm_{args.model}.csv')
    new = not os.path.exists(out_csv)
    fout = open(out_csv, 'a')
    if new:
        fout.write('model,epoch,space,budget,stat,seed,T,p\n')

    def emit(ep, space, budget, stat, seed, T, p):
        fout.write(f'{args.model},{ep},{space},{budget},{stat},{seed},'
                   f'{T:+.6e},{p:.6f}\n')
        fout.flush()

    for ep in eps:
        t0 = time.time()
        z = np.load(os.path.join(args.feats_dir, f'ep{ep:03d}.npz'))
        h_tr = z['h_tr'].astype(np.float64)
        h_te = z['h_te'].astype(np.float64)
        D = h_tr.shape[1]
        spaces = {'unnorm': (h_tr, h_te), 'l2': (l2n(h_tr), l2n(h_te))}

        # ── MMD, both spaces, both budgets ──
        for sp, (a_tr, a_te) in spaces.items():
            for seed in range(SEED0, SEED0 + args.mmd_seeds):
                rng = np.random.RandomState(seed)
                # 1500 tier (paper's baseline budget)
                X = a_tr[rng.permutation(len(a_tr))[:1500]]
                Y = a_te[rng.permutation(len(a_te))[:1500]]
                _, p = mmd_fast(X, Y, args.nperm_small, seed=seed)
                emit(ep, sp, 'b1500', 'mmd', seed, np.nan, p)
                # 10k equal-budget tier
                if ep % args.mmd10k_every == 0:
                    X = a_tr[rng.permutation(len(a_tr))[:10000]]
                    Y = a_te[rng.permutation(len(a_te))[:10000]]
                    _, p = mmd_fast(X, Y, args.nperm_big, seed=seed)
                    emit(ep, sp, 'b10k', 'mmd', seed, np.nan, p)

        # ── Euclidean per-point ladder on unnorm space ──
        if not args.skip_eucl:
            for bname, B in BUDGETS.items():
                for seed in range(SEED0, SEED0 + SEEDS_POINT):
                    rng = np.random.RandomState(seed)
                    perm = rng.permutation(len(h_tr))
                    ref = h_tr[perm[:B['M']]]
                    qtr = h_tr[perm[B['M']:B['M'] + B['n_qtr']]]
                    if B['n_qte'] is None:
                        qte = h_te
                    else:
                        qte = h_te[rng.permutation(len(h_te))[:B['n_qte']]]
                    k = min(K, B['M'])
                    s_tr = eucl_scores(topk_eucl(qtr, ref, k), D)
                    s_te = eucl_scores(topk_eucl(qte, ref, k), D)
                    for stat in ('dist_topk', 'r_k', 'logV_k', 'chat_eucl'):
                        T, p = perm_test_med(s_tr[stat], s_te[stat], seed=seed)
                        emit(ep, 'unnorm', bname, stat, seed, T, p)

        print(f'ep{ep:03d} done in {time.time()-t0:.1f}s', flush=True)

    fout.close()
    print('->', out_csv)


if __name__ == '__main__':
    main()
