"""
ksweep_low.py — k sensitivity of theta_k / chat at the LOW budget tier
======================================================================
Diagnosis run: at the low budget (M=500 ref + 1000 train queries + 1500
test queries, total-access matched to the 1500-per-split baseline
budget), the full VRT score (chat, k=60) loses power while theta_k
retains it. Hypothesis: k=60 is mismatched to M=500 (k/M = 12%); deep-
rank spacings are noise-dominated. This script sweeps k at the low
budget only, for theta_k and chat, under the exact ladder protocol
(same splits, same median statistic, same permutation test, seeds
42..51).

Cost trick: top-60 cosines are computed once per (epoch, seed); every
smaller k is a prefix slice of the sorted top-k matrix.

Usage:
  python ksweep_low.py --model r50 --feats_dir <...>/ladder_feats/r50 \
      --out_dir <...>/ladder_out/ksweep_r50 --epochs 50-130 --every 5
Output: CSV <out>/ksweep_{model}.csv
  columns: model,epoch,k,stat,seed,T,p
"""
import argparse
import os
import re
import time

import numpy as np
from scipy.special import betainc, betaln

SEED0 = 42
N_PERM = 2000
KS = (15, 30, 45, 60)
KMAX = max(KS)

M, N_QTR, N_QTE = 500, 1000, 1500          # low budget tier (fixed)


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


def scores_for_k(top, D, k):
    theta = np.arccos(np.clip(top[:, :k], -1, 1))
    L = log_cap_volume(theta, D)
    dL = np.diff(L, axis=1)
    r = np.arange(1, k, dtype=np.float64)
    Z = np.maximum(r[None, :] * dL, 0.0)
    return {
        'theta_k': theta[:, -1],
        'chat': np.maximum(Z.mean(axis=1), 1e-12),
    }


def perm_test(s_tr, s_te, n_perm=N_PERM, seed=0):
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


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def parse_epochs(spec, available, every):
    if spec is None:
        keep = available
    else:
        s = set()
        for part in spec.split(','):
            if '-' in part:
                a, b = part.split('-')
                s.update(range(int(a), int(b)+1))
            else:
                s.add(int(part))
        keep = [e for e in available if e in s]
    return [e for e in keep if e % every == 0] or keep


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
    eps = parse_epochs(args.epochs, [int(f[2:-4]) for f in files], args.every)
    print(f'{args.model}: {len(eps)} epochs, ks={KS}', flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f'ksweep_{args.model}.csv')
    new = not os.path.exists(out_csv)
    fout = open(out_csv, 'a')
    if new:
        fout.write('model,epoch,k,stat,seed,T,p\n')

    done = set()
    if not new:
        import csv as _csv
        for r in _csv.reader(open(out_csv)):
            if r and r[0] == args.model:
                done.add((int(r[1]), int(r[2]), r[3], int(r[4])))

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
            qte = e_te[rng.permutation(len(e_te))[:N_QTE]]
            top_tr = topk_cosines(qtr, ref, KMAX)
            top_te = topk_cosines(qte, ref, KMAX)
            for k in KS:
                s_tr = scores_for_k(top_tr, D, k)
                s_te = scores_for_k(top_te, D, k)
                for stat in ('theta_k', 'chat'):
                    if (ep, k, stat, seed) in done:
                        continue
                    T, p = perm_test(s_tr[stat], s_te[stat], seed=seed)
                    fout.write(f'{args.model},{ep},{k},{stat},{seed},'
                               f'{T:+.6e},{p:.6f}\n')
        fout.flush()
        print(f'ep{ep:03d} done in {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
