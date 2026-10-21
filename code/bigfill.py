"""bigfill.py -- fill non-grid epochs (ep % 5 != 0) for ONE big-budget
method on ONE model unit. Writes its own fill CSV."""
import argparse, os, re
import numpy as np
from baselines2_stats import l2n, c2st_test, ed_cached, mmdagg_test

ap = argparse.ArgumentParser()
ap.add_argument('--method', required=True, choices=['c2st', 'ed', 'mmdagg'])
ap.add_argument('--unit', required=True)
ap.add_argument('--feats_dir', required=True)
ap.add_argument('--epochs', required=True)
ap.add_argument('--out_dir', required=True)
a = ap.parse_args()

lo, hi = map(int, a.epochs.split('-'))
eps = sorted(int(f[2:-4]) for f in os.listdir(a.feats_dir)
             if re.fullmatch(r'ep\d+\.npz', f))
eps = [e for e in eps if lo <= e <= hi and e % 5 != 0]
os.makedirs(a.out_dir, exist_ok=True)
out = os.path.join(a.out_dir, f'fill_{a.method}_{a.unit}.csv')
f = open(out, 'w')
f.write('model,epoch,budget,stat,seed,acc,p\n')

for ep in eps:
    z = np.load(os.path.join(a.feats_dir, f'ep{ep:03d}.npz'))
    e_tr = l2n(z['h_tr'].astype(np.float64))
    e_te = l2n(z['h_te'].astype(np.float64))
    if a.method == 'c2st':
        for seed in range(42, 47):
            acc, p = c2st_test(e_tr, e_te, n_sub=10000, seed=seed)
            f.write(f'{a.unit},{ep},b10k,c2st,{seed},{acc:.6f},{p:.6f}\n')
    elif a.method == 'ed':
        for seed in range(42, 47):
            _, p = ed_cached(e_tr, e_te, 10000, 200, seed=seed)
            f.write(f'{a.unit},{ep},b10k,ed,{seed},nan,{p:.6f}\n')
    else:
        for seed in (42, 43):
            try:
                rej = mmdagg_test(e_tr, e_te, n_sub=5000, seed=seed)
                f.write(f'{a.unit},{ep},b5000,mmdagg,{seed},nan,{float(rej):.6f}\n')
            except Exception as exc:
                print(f'mmdagg fail ep{ep}: {exc}', flush=True)
    f.flush()
    print(f'ep{ep} done', flush=True)
f.close()
print('->', out)
