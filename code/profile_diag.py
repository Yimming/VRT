"""
profile_diag.py — rank-profile mechanism diagnostic (no permutations)
=====================================================================
For chosen epochs, computes the full kNN log-volume curve per query and
reports, for each rank r = 1..k:

    med_Ltr(r) = median over train queries of L_r
    med_Lte(r) = median over test  queries of L_r
    delta(r)   = med_Lte(r) - med_Ltr(r)

Identity: chat = L_k - mean(L_1..L_{k-1}), so
  * delta(r) ~ constant in r  -> endpoint/shift-type alternative
                                 (theta_k sensitive, chat cancels it)
  * delta(r) varying in r     -> profile/shape-type alternative
                                 (chat sensitive, theta_k blind to interior)

Protocol matches the ladder paper budget (M=2000 ref, 8000 train
queries, all test queries), seeds 42..46 averaged.

Usage:
  python profile_diag.py --model zoo_c10_m0 --feats_dir <...> \
      --out_dir <...> --epochs 20,25,30,40,59
Output: CSV <out>/profile_{model}.csv
  columns: model,epoch,r,med_Ltr,med_Lte,delta
"""
import argparse
import os

import numpy as np
from scipy.special import betainc, betaln

K = 60
M, N_QTR = 2000, 8000
SEEDS = range(42, 47)          # 5 seeds, averaged


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
    ap.add_argument('--epochs', required=True, help='comma list, e.g. 20,30,59')
    args = ap.parse_args()

    eps = [int(e) for e in args.epochs.split(',')]
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f'profile_{args.model}.csv')
    fout = open(out_csv, 'w')
    fout.write('model,epoch,r,med_Ltr,med_Lte,delta\n')

    for ep in eps:
        path = os.path.join(args.feats_dir, f'ep{ep:03d}.npz')
        if not os.path.exists(path):
            print(f'ep{ep}: npz missing, skipped', flush=True)
            continue
        z = np.load(path)
        e_tr = l2n(z['h_tr'].astype(np.float64))
        e_te = l2n(z['h_te'].astype(np.float64))
        D = e_tr.shape[1]
        acc_tr = np.zeros(K)
        acc_te = np.zeros(K)
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(e_tr))
            ref = e_tr[perm[:M]]
            qtr = e_tr[perm[M:M+N_QTR]]
            L_tr = log_cap_volume(np.arccos(np.clip(
                topk_cosines(qtr, ref, K), -1, 1)), D)
            L_te = log_cap_volume(np.arccos(np.clip(
                topk_cosines(e_te, ref, K), -1, 1)), D)
            acc_tr += np.median(L_tr, axis=0)
            acc_te += np.median(L_te, axis=0)
        acc_tr /= len(list(SEEDS))
        acc_te /= len(list(SEEDS))
        for r in range(K):
            fout.write(f'{args.model},{ep},{r+1},{acc_tr[r]:+.6f},'
                       f'{acc_te[r]:+.6f},{acc_te[r]-acc_tr[r]:+.6f}\n')
        fout.flush()
        d = acc_te - acc_tr
        print(f'ep{ep:03d}: delta(r=1)={d[0]:+.4f} delta(r=30)={d[29]:+.4f} '
              f'delta(r=60)={d[59]:+.4f}  range={d.max()-d.min():.4f}', flush=True)


if __name__ == '__main__':
    main()
