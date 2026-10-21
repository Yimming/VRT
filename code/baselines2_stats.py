"""
baselines2_stats.py — MMDAgg / C2ST / Energy Distance at two budgets
====================================================================
Completes the fair-budget table (reviewer Ftg8) for the remaining paper
baselines, using the supplementary methods.py implementations:

  mmdagg : verbatim (mmdagg.np package; binary reject decision)
  c2st   : verbatim (sklearn MLP, 20 ReLU units, normal-approx p; the
           paper's additional acc>0.55 convention is applied at summary
           time, both acc and p are recorded here)
  ed     : same statistic as methods.py ed_stat (biased /n^2 with zero
           diagonal), computed from a CACHED pooled distance matrix with
           matvec permutations — mathematically identical, fast enough
           to run at D=2048 and n=10k (the naive version is not)

Budgets:
  small : n_sub = 1500 per split  (paper's baseline budget; C2ST also
          run at its paper value 8000 via --c2st_paper)
  big   : n_sub = 10000 per split (equal to VRT's total access)
          MMDAgg big tier uses --mmdagg_big_n (default 5000, the largest
          tractable size for the numpy implementation; noted in rebuttal)

All tests run on l2-normalized features (paper protocol).

Output CSV: base2_{model}.csv with
  model,epoch,budget,stat,seed,acc,p
  (acc is NaN except for c2st; for mmdagg, p is 0/1 reject flag)

Usage:
  python baselines2_stats.py --model v7  --feats_dir <...>/v7
  python baselines2_stats.py --model r50 --feats_dir <...>/r50 --epochs 0-50
"""
import argparse
import os
import re
import time

import numpy as np
from scipy.stats import norm as norm_dist
from sklearn.neural_network import MLPClassifier

try:
    from mmdagg.np import mmdagg as mmdagg_fn
except ImportError:
    mmdagg_fn = None


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


# ── C2ST (verbatim from methods.py) ───────────────────────────────────
def c2st_test(emb_tr, emb_te, n_sub=8000, seed=42):
    rng = np.random.RandomState(seed)
    n_each = min(n_sub, len(emb_tr), len(emb_te))
    X_p = emb_tr[rng.permutation(len(emb_tr))[:n_each]]
    X_q = emb_te[rng.permutation(len(emb_te))[:n_each]]
    X = np.concatenate([X_p, X_q])
    if not np.isfinite(X).all():                             # release fix: sklearn would raise or fit garbage silently
        raise ValueError("c2st_test: the subsample contains NaN or inf")
    y = np.array([0] * n_each + [1] * n_each)
    shuf = rng.permutation(len(X))
    X, y = X[shuf], y[shuf]
    n_tr = len(X) // 2
    X_train, y_train = X[:n_tr], y[:n_tr]
    X_test, y_test = X[n_tr:], y[n_tr:]
    n_te = len(X_test)
    clf = MLPClassifier(hidden_layer_sizes=(20,), activation='relu',
                        solver='adam', max_iter=100, random_state=seed)
    clf.fit(X_train, y_train)
    acc = float((clf.predict(X_test) == y_test).mean())
    se = np.sqrt(1.0 / (4.0 * n_te))
    z = (acc - 0.5) / se
    return acc, float(1.0 - norm_dist.cdf(z))


# ── MMDAgg (verbatim protocol from methods.py) ────────────────────────
def mmdagg_test(emb_tr, emb_te, n_sub=1500, seed=42):
    if mmdagg_fn is None:
        raise ImportError("mmdagg is required. Install with: pip install mmdagg")
    rng = np.random.RandomState(seed)
    X = emb_tr[rng.permutation(len(emb_tr))[:n_sub]].astype(np.float32)
    Y = emb_te[rng.permutation(len(emb_te))[:n_sub]].astype(np.float32)
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):   # release fix: the package compares NaN statistics silently
        raise ValueError("mmdagg_test: the subsample contains NaN or inf")
    return int(mmdagg_fn(X, Y, seed=seed))


# ── Energy Distance, cached-distance implementation ───────────────────
def ed_cached(emb_tr, emb_te, n_sub, n_perm, seed=42):
    """Same statistic as methods.py ed_stat: 2*mean(dxy) - mean(dxx) -
    mean(dyy), Euclidean distances, biased /n^2 normalization with zero
    diagonal. Pooled distance matrix computed once; permutations are one
    matvec each."""
    rng = np.random.RandomState(seed)
    X = emb_tr[rng.permutation(len(emb_tr))[:n_sub]]
    Y = emb_te[rng.permutation(len(emb_te))[:n_sub]]
    pooled = np.concatenate([X, Y]).astype(np.float64)
    n, m = len(X), len(Y)
    N = n + m
    sq = np.sum(pooled**2, 1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2 * pooled @ pooled.T, 0.0)
    Dm = np.sqrt(d2, out=d2).astype(np.float32)
    np.fill_diagonal(Dm, 0.0)
    tot = float(Dm.sum())

    def stat(z):
        S = Dm @ z
        xx = float(z @ S)
        sx = float(S.sum())
        xy = sx - xx
        yy = tot - xx - 2 * xy
        return (n * m / (n + m)) * (2 * xy / (n * m) - xx / (n * n) - yy / (m * m))

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
            keep.update(range(int(a), int(b) + 1))
        else:
            keep.add(int(part))
    return [e for e in available if e in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--feats_dir', required=True)
    ap.add_argument('--out_dir', default='.')
    ap.add_argument('--epochs', default=None)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--big_every', type=int, default=5,
                    help='run big-budget tiers only every N epochs')
    ap.add_argument('--mmdagg_big_n', type=int, default=5000)
    ap.add_argument('--c2st_paper', action='store_true',
                    help='also run C2ST at its paper budget n=8000')
    ap.add_argument('--skip_mmdagg', action='store_true')
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.feats_dir)
                   if re.fullmatch(r'ep\d+\.npz', f))
    eps = parse_epochs(args.epochs, [int(f[2:-4]) for f in files])
    print(f'{args.model}: {len(eps)} epochs; mmdagg available: {mmdagg_fn is not None}')

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f'base2_{args.model}.csv')
    new = not os.path.exists(out_csv)
    fout = open(out_csv, 'a')
    if new:
        fout.write('model,epoch,budget,stat,seed,acc,p\n')

    def emit(ep, budget, stat, seed, acc, p):
        fout.write(f'{args.model},{ep},{budget},{stat},{seed},{acc:.6f},{p:.6f}\n')
        fout.flush()

    for ep in eps:
        t0 = time.time()
        z = np.load(os.path.join(args.feats_dir, f'ep{ep:03d}.npz'))
        e_tr = l2n(z['h_tr'].astype(np.float64))
        e_te = l2n(z['h_te'].astype(np.float64))
        big = (ep % args.big_every == 0)

        for seed in range(42, 42 + args.seeds):   # paper seed convention
            # ── ED ──
            _, p = ed_cached(e_tr, e_te, 1500, 500, seed=seed)
            emit(ep, 'b1500', 'ed', seed, np.nan, p)
            if big:
                _, p = ed_cached(e_tr, e_te, 10000, 200, seed=seed)
                emit(ep, 'b10k', 'ed', seed, np.nan, p)

            # ── C2ST ──
            acc, p = c2st_test(e_tr, e_te, n_sub=1500, seed=seed)
            emit(ep, 'b1500', 'c2st', seed, acc, p)
            if args.c2st_paper:
                acc, p = c2st_test(e_tr, e_te, n_sub=8000, seed=seed)
                emit(ep, 'b8000', 'c2st', seed, acc, p)
            if big:
                acc, p = c2st_test(e_tr, e_te, n_sub=10000, seed=seed)
                emit(ep, 'b10k', 'c2st', seed, acc, p)

            # ── MMDAgg ──
            if not args.skip_mmdagg and mmdagg_fn is not None:
                rej = mmdagg_test(e_tr, e_te, n_sub=1500, seed=seed)
                emit(ep, 'b1500', 'mmdagg', seed, np.nan, float(rej))
                if big and seed < 44:
                    try:
                        rej = mmdagg_test(e_tr, e_te,
                                          n_sub=args.mmdagg_big_n, seed=seed)
                        emit(ep, f'b{args.mmdagg_big_n}', 'mmdagg', seed,
                             np.nan, float(rej))
                    except (MemoryError, Exception) as exc:  # noqa: BLE001
                        print(f'mmdagg big failed at ep{ep}: {exc}', flush=True)

        print(f'ep{ep:03d} done in {time.time()-t0:.1f}s', flush=True)

    fout.close()
    print('->', out_csv)


if __name__ == '__main__':
    main()
