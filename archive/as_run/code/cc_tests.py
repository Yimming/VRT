"""
cc_tests.py — class-conditional independence tests (VRT-independent), v2
========================================================================
Independent evidence for nxR4 Q5. Null hypothesis, per fixed checkpoint:

  H0: P(Z | Y=c, S=train) = P(Z | Y=c, S=test)  for every class c
      (within-class exchangeability of train/test membership,
       conditional on the pooled embeddings)

Two pre-registered statistics on l2-normalized embeddings, calibrated by
the SAME within-class permutations, combined into a max-T omnibus:

  T_mean   : class-averaged unbiased linear-kernel MMD^2
             (Gretton et al. 2012, Lemma 6 with k(x,y)=x'y; population
             value ||mu_tr,c - mu_te,c||^2; estimate may be negative —
             evidence comes from the permutation p, not the sign)
  T_spread : class-averaged two-sample Cramer-von-Mises on angular
             residuals r_i = 1 - u_c' z_i, where u_c is the POOLED class
             mean direction (permutation-invariant). Sensitive to
             spread/tail/multimodality changes even when means coincide.
  omnibus  : max of permutation-standardized (T_mean, T_spread); p from
             the permutation distribution of the max (Westfall-Young
             style, same permutation set). One conclusion per setting.

Rejection (after Holm over the pre-registered settings) certifies
independently corroborated CLASS-CONDITIONAL backbone divergence.
It does not certify onset epochs, and non-rejection is not proof of
absence.

Usage:
  python cc_tests.py --model ablC --dataset cifar10 --feats_dir <...>/ablC \
      --out_dir <...> --epochs 49 --n_perm 9999
"""
import argparse
import os
import re

import numpy as np
import torchvision

DATA_FOLDER = '/home3/p318770/venvs/overfitting'


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def linear_mmd2_unbiased(X, Y):
    n, m = len(X), len(Y)
    Sx = X.sum(axis=0)
    Sy = Y.sum(axis=0)
    sqx = float(np.einsum('ij,ij->', X, X))
    sqy = float(np.einsum('ij,ij->', Y, Y))
    return ((float(Sx @ Sx) - sqx) / (n * (n - 1))
            + (float(Sy @ Sy) - sqy) / (m * (m - 1))
            - 2.0 * float(Sx @ Sy) / (n * m))


def cvm_from_mask(mask_sorted, n, m):
    """Two-sample Cramer-von-Mises from a boolean membership array aligned
    to the pooled SORTED residuals (True = train). O(N)."""
    N = n + m
    ca = np.cumsum(mask_sorted)
    Fa = ca / n
    Fb = (np.arange(1, N + 1) - ca) / m
    return n * m / (N * N) * float(np.sum((Fa - Fb) ** 2))


def test_epoch(e_tr, e_te, y_tr, y_te, n_classes, n_perm, seed=0,
               chunk=512):
    rng = np.random.RandomState(seed)
    # per-class precomputation (all permutation-invariant pieces)
    cls_data = []
    for c in range(n_classes):
        Xc = e_tr[y_tr == c]
        Yc = e_te[y_te == c]
        P = np.concatenate([Xc, Yc])
        n, m = len(Xc), len(Yc)
        u = P.sum(axis=0)
        u = u / np.linalg.norm(u)                    # pooled mean direction
        r = 1.0 - P @ u                              # angular residuals
        order = np.argsort(r, kind='mergesort')      # fixed sort, reused
        is_train = np.zeros(n + m, dtype=bool)
        is_train[:n] = True
        q = np.einsum('ij,ij->i', P, P)              # per-sample sq norms
        cls_data.append(dict(P=P, n=n, m=m, order=order, is_train=is_train,
                             q=q, St=P.sum(axis=0), qt=float(q.sum())))

    def stats_single(assignments):
        Us, Cs = [], []
        for c, d in enumerate(cls_data):
            a = assignments[c]
            X = d['P'][a]
            Y = d['P'][~a]
            Us.append(linear_mmd2_unbiased(X, Y))
            Cs.append(cvm_from_mask(a[d['order']], d['n'], d['m']))
        return float(np.mean(Us)), float(np.mean(Cs))

    obs = stats_single([d['is_train'] for d in cls_data])

    # ── vectorized permutations: a chunk of B masks at once per class ──
    perm_stats = np.empty((n_perm, 2))
    done = 0
    while done < n_perm:
        B = min(chunk, n_perm - done)
        U_acc = np.zeros(B)
        C_acc = np.zeros(B)
        for d in cls_data:
            N, n, m = d['n'] + d['m'], d['n'], d['m']
            # B random train-masks via argpartition of uniform noise
            R = rng.random((B, N))
            idx = np.argpartition(R, n - 1, axis=1)[:, :n]
            A = np.zeros((B, N))
            np.put_along_axis(A, idx, 1.0, axis=1)
            # linear MMD, all B perms in one BLAS call
            S = A @ d['P']                       # (B, D) train-group sums
            sq = A @ d['q']                      # (B,)  train sq-norm sums
            Sy = d['St'][None, :] - S
            sqy = d['qt'] - sq
            Sx2 = np.einsum('ij,ij->i', S, S)
            Sy2 = np.einsum('ij,ij->i', Sy, Sy)
            SxSy = np.einsum('ij,ij->i', S, Sy)
            U = ((Sx2 - sq) / (n * (n - 1))
                 + (Sy2 - sqy) / (m * (m - 1))
                 - 2.0 * SxSy / (n * m))
            # CvM, vectorized over the chunk
            As = A[:, d['order']]
            ca = np.cumsum(As, axis=1)
            Fa = ca / n
            Fb = (np.arange(1, N + 1)[None, :] - ca) / m
            C = n * m / (N * N) * np.sum((Fa - Fb) ** 2, axis=1)
            U_acc += U
            C_acc += C
        perm_stats[done:done + B, 0] = U_acc / n_classes
        perm_stats[done:done + B, 1] = C_acc / n_classes
        done += B

    # marginal one-sided permutation p-values
    p_mean = (int(np.sum(perm_stats[:, 0] >= obs[0])) + 1) / (n_perm + 1)
    p_spread = (int(np.sum(perm_stats[:, 1] >= obs[1])) + 1) / (n_perm + 1)

    # max-T omnibus: standardize each statistic by its permutation
    # mean/std, take the max, compare with the permutation maxima
    mu = perm_stats.mean(axis=0)
    sd = np.maximum(perm_stats.std(axis=0), 1e-30)
    z_obs = (np.array(obs) - mu) / sd
    z_perm = (perm_stats - mu) / sd
    max_obs = float(z_obs.max())
    max_perm = z_perm.max(axis=1)
    p_omni = (int(np.sum(max_perm >= max_obs)) + 1) / (n_perm + 1)

    return obs[0], p_mean, obs[1], p_spread, p_omni


def parse_epochs(spec, available):
    if spec is None:
        return available
    keep = set()
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            keep.update(range(int(a), int(b) + 1))
        else:
            keep.add(int(part))
    return [e for e in available if e in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--dataset', choices=['cifar10', 'cifar100'], required=True)
    ap.add_argument('--feats_dir', required=True)
    ap.add_argument('--out_dir', default='.')
    ap.add_argument('--epochs', default=None)
    ap.add_argument('--n_perm', type=int, default=999)
    args = ap.parse_args()

    cls = (torchvision.datasets.CIFAR10 if args.dataset == 'cifar10'
           else torchvision.datasets.CIFAR100)
    y_tr = np.array(cls(DATA_FOLDER, True, download=False).targets)
    y_te = np.array(cls(DATA_FOLDER, False, download=False).targets)
    n_classes = 10 if args.dataset == 'cifar10' else 100

    files = sorted(f for f in os.listdir(args.feats_dir)
                   if re.fullmatch(r'ep\d+\.npz', f))
    eps = parse_epochs(args.epochs, [int(f[2:-4]) for f in files])

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f'cctests_{args.model}.csv')
    new = not os.path.exists(out_csv)
    fout = open(out_csv, 'a')
    if new:
        fout.write('model,epoch,n_perm,T_mean,p_mean,T_spread,p_spread,p_omnibus\n')

    for ep in eps:
        z = np.load(os.path.join(args.feats_dir, f'ep{ep:03d}.npz'))
        e_tr = l2n(z['h_tr'].astype(np.float64))
        e_te = l2n(z['h_te'].astype(np.float64))
        Tm, pm, Ts, ps, po = test_epoch(e_tr, e_te, y_tr, y_te, n_classes,
                                        args.n_perm, seed=ep)
        fout.write(f'{args.model},{ep},{args.n_perm},'
                   f'{Tm:+.6e},{pm:.6f},{Ts:+.6e},{ps:.6f},{po:.6f}\n')
        fout.flush()
        print(f'ep{ep:03d}: mean T={Tm:+.3e} p={pm:.4f} | '
              f'spread T={Ts:+.3e} p={ps:.4f} | omnibus p={po:.4f}',
              flush=True)
    fout.close()
    print('->', out_csv)


if __name__ == '__main__':
    main()
