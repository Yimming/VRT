"""
decomp_stats.py — fine-grained decomposition ladder between the
reviewer's statistic (mean cosine) and the full VRT score.
=================================================================
All statistics are computed from the SAME top-k=60 neighborhood
matrix, same reference/query splits, same median-difference
permutation test, paper budget (M=2000, 8000 train queries, all
test queries), seeds 42..51.

  cosmean      : mean cosine to top-k neighbors        (reviewer's)
  thetamean    : mean angle over ranks 1..k            (cos -> angle)
  logthetamean : mean log-angle over ranks 1..k        (+ log)
  Lmean        : mean log-cap-volume over ranks 1..k   (+ exact volume map)
  thetak       : angle at rank k only                  (single order stat)
  chat         : L_k - mean(L_1..L_{k-1})              (level -> contrast)

Usage:
  python decomp_stats.py --model r50 --feats_dir <...>/r50 \
      --out_dir <...>/decomp_r50 --epochs 40-140 --every 2
Output CSV: model,epoch,stat,seed,T,p
"""
import argparse
import os
import re
import time

import numpy as np
from scipy.special import betainc, betaln

K = 60
M, N_QTR = 2000, 8000
N_PERM = 2000
SEED0 = 42


def log_cap_volume(theta, D):
    sin2 = np.sin(theta)**2
    I = betainc((D-1)/2.0, 0.5, sin2)
    V = np.where(theta <= np.pi/2, 0.5*I, 1.0 - 0.5*I)
    safe = V > 1e-300
    out = np.empty_like(V)
    out[safe] = np.log(V[safe])
    if np.any(~safe):
        a = (D-1)/2.0
        const = -np.log(2.0) - np.log(a) - betaln(a, 0.5)
        out[~safe] = const + (D-1)*np.log(np.maximum(np.sin(theta[~safe]), 1e-300))
    return out


def topk(query, ref, k, batch=2048):
    out = []
    for s in range(0, query.shape[0], batch):
        cos = query[s:s+batch] @ ref.T
        pidx = np.argpartition(-cos, k-1, axis=1)[:, :k]
        top = np.take_along_axis(cos, pidx, axis=1)
        top = np.take_along_axis(top, np.argsort(-top, axis=1), axis=1)
        out.append(top)
    return np.concatenate(out, 0)


def all_scores(top, D, k=K):
    theta = np.arccos(np.clip(top, -1, 1))
    L = log_cap_volume(theta, D)
    dL = np.diff(L, axis=1)
    r = np.arange(1, k, dtype=float)
    Z = np.maximum(r[None, :]*dL, 0.0)
    return {
        'cosmean': top.mean(1),
        'thetamean': theta.mean(1),
        'logthetamean': np.log(np.maximum(theta, 1e-12)).mean(1),
        'Lmean': L.mean(1),
        'thetak': theta[:, -1],
        'chat': np.maximum(Z.mean(1), 1e-12),
    }


def perm_p(a, b, rng, n_perm=N_PERM):
    obs = np.median(b) - np.median(a)
    pooled = np.concatenate([a, b]); na = len(a); cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled))
        if abs(np.median(pooled[p[na:]]) - np.median(pooled[p[:na]])) >= abs(obs):
            cnt += 1
    return obs, (cnt+1)/(n_perm+1)


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--feats_dir', required=True)
    ap.add_argument('--out_dir', default='.')
    ap.add_argument('--epochs', default=None)
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--seeds', type=int, default=10)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.feats_dir)
                   if re.fullmatch(r'ep\d+\.npz', f))
    eps = [int(f[2:-4]) for f in files]
    if args.epochs:
        a, b = args.epochs.split('-')
        eps = [e for e in eps if int(a) <= e <= int(b)]
    eps = [e for e in eps if e % args.every == 0] or eps
    print(f'{args.model}: {len(eps)} epochs', flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f'decomp_{args.model}.csv')
    done = set()
    if os.path.exists(out_csv):
        for r in __import__('csv').reader(open(out_csv)):
            if r and r[0] == args.model:
                done.add((int(r[1]), r[2], int(r[3])))
    out = open(out_csv, 'a')
    if out.tell() == 0:
        out.write('model,epoch,stat,seed,T,p\n')

    for ep in eps:
        t0 = time.time()
        z = np.load(os.path.join(args.feats_dir, f'ep{ep:03d}.npz'))
        e_tr = l2n(z['h_tr'].astype(np.float64))
        e_te = l2n(z['h_te'].astype(np.float64))
        D = e_tr.shape[1]
        for seed in range(SEED0, SEED0 + args.seeds):
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(e_tr))
            ref = e_tr[perm[:M]]
            qtr = e_tr[perm[M:M+N_QTR]]
            s_tr = all_scores(topk(qtr, ref, K), D)
            s_te = all_scores(topk(e_te, ref, K), D)
            for stat in ('cosmean', 'thetamean', 'logthetamean',
                         'Lmean', 'thetak', 'chat'):
                if (ep, stat, seed) in done:
                    continue
                T, p = perm_p(s_tr[stat], s_te[stat], rng)
                out.write(f'{args.model},{ep},{stat},{seed},'
                          f'{T:+.6e},{p:.6f}\n')
        out.flush()
        print(f'ep{ep:03d} done in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
