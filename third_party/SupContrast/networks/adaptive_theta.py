"""
adaptive_theta.py — rank-adaptive single-rank test vs chat, shared perms
========================================================================
Legitimate multi-rank competitor for chat built ONLY from single-rank
statistics: for every rank r = 2..60, T_r = median-diff of theta_r; each
rank is standardized by its own permutation null (mean/std over the SAME
B permutations); the adaptive statistic is M = max_r |Z_r|, calibrated
against the permutation distribution of max_r |Z_r^(b)| (Westfall-
Troendle resampling max-statistic). This pays the multiple-comparison
cost of searching 59 ranks. chat is tested with the IDENTICAL
permutations for a same-noise head-to-head.

Paper budget (M=2000 ref + 8000 qtr + all test), seeds 42..46.

Usage:
  python adaptive_theta.py --model r50 --feats_dir <...>/r50 \
      --out_dir <...>/adapt_r50 --epochs 50-75
Output CSV: model,epoch,seed,p_adaptive,best_rank,p_theta60,p_chat
"""
import argparse
import os
import re
import time

import numpy as np
from scipy.special import betainc, betaln

K = 60
M, N_QTR = 2000, 8000
N_PERM = 1000
SEEDS = range(42, 47)


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


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--feats_dir', required=True)
    ap.add_argument('--out_dir', default='.')
    ap.add_argument('--epochs', default=None, help='e.g. 50-75')
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.feats_dir)
                   if re.fullmatch(r'ep\d+\.npz', f))
    eps = [int(f[2:-4]) for f in files]
    if args.epochs:
        a, b = args.epochs.split('-')
        eps = [e for e in eps if int(a) <= e <= int(b)]
    print(f'{args.model}: {len(eps)} epochs', flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out = open(os.path.join(args.out_dir, f'adapt_{args.model}.csv'), 'a')
    if out.tell() == 0:
        out.write('model,epoch,seed,p_adaptive,best_rank,p_theta60,p_chat\n')
    outr = open(os.path.join(args.out_dir, f'adapt_ranks_{args.model}.csv'), 'a')
    if outr.tell() == 0:
        outr.write('model,epoch,seed,r,p\n')

    for ep in eps:
        t0 = time.time()
        z = np.load(os.path.join(args.feats_dir, f'ep{ep:03d}.npz'))
        e_tr = l2n(z['h_tr'].astype(np.float64))
        e_te = l2n(z['h_te'].astype(np.float64))
        D = e_tr.shape[1]
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(e_tr))
            ref = e_tr[perm[:M]]
            qtr = e_tr[perm[M:M+N_QTR]]
            th_tr = np.arccos(np.clip(topk(qtr, ref, K), -1, 1))
            th_te = np.arccos(np.clip(topk(e_te, ref, K), -1, 1))
            # per-rank theta matrices (use ranks 2..60 -> cols 1..59)
            S_tr = th_tr[:, 1:]                     # (n_tr, 59)
            S_te = th_te[:, 1:]
            # chat from the same neighborhoods
            L_tr = log_cap_volume(th_tr, D); L_te = log_cap_volume(th_te, D)
            r = np.arange(1, K, dtype=float)
            c_tr = np.maximum((r*np.diff(L_tr, axis=1)).mean(1), 1e-12)
            c_te = np.maximum((r*np.diff(L_te, axis=1)).mean(1), 1e-12)

            pool = np.concatenate([S_tr, S_te])     # (n, 59)
            poolc = np.concatenate([c_tr, c_te])
            na, n = len(S_tr), len(pool)
            obs_r = np.median(S_te, 0) - np.median(S_tr, 0)      # (59,)
            obs_c = np.median(c_te) - np.median(c_tr)

            Tb = np.empty((N_PERM, 59)); Tcb = np.empty(N_PERM)
            for b in range(N_PERM):
                p = rng.permutation(n)
                Tb[b] = np.median(pool[p[na:]], 0) - np.median(pool[p[:na]], 0)
                Tcb[b] = np.median(poolc[p[na:]]) - np.median(poolc[p[:na]])
            mu, sd = Tb.mean(0), np.maximum(Tb.std(0), 1e-12)
            Zb = np.abs((Tb - mu)/sd)
            Zo = np.abs((obs_r - mu)/sd)
            Mb = Zb.max(1)
            Mo = Zo.max()
            p_adapt = (1 + np.sum(Mb >= Mo)) / (N_PERM + 1)
            best = int(np.argmax(Zo)) + 2
            p_t60 = (1 + np.sum(np.abs(Tb[:, -1]) >= abs(obs_r[-1]))) / (N_PERM + 1)
            p_chat = (1 + np.sum(np.abs(Tcb) >= abs(obs_c))) / (N_PERM + 1)
            out.write(f'{args.model},{ep},{seed},{p_adapt:.6f},{best},'
                      f'{p_t60:.6f},{p_chat:.6f}\n')
            # per-rank p-values: rank-selection fragility curve
            p_ranks = (1 + np.sum(np.abs(Tb) >= np.abs(obs_r)[None, :], axis=0)) \
                      / (N_PERM + 1)
            for j, pr in enumerate(p_ranks):
                outr.write(f'{args.model},{ep},{seed},{j+2},{pr:.6f}\n')
            outr.flush()
        out.flush()
        print(f'ep{ep:03d} done in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
