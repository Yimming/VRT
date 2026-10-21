"""bigfill2.py -- targeted refill: compute ONLY the explicitly listed
epochs for one method on one unit. Writes fill2_{method}_{unit}.csv
(never touches existing fill files; harvest unions everything)."""
import argparse, os
import numpy as np
from baselines2_stats import l2n, c2st_test, ed_cached, mmdagg_test

ap = argparse.ArgumentParser()
ap.add_argument('--method', required=True, choices=['c2st', 'ed', 'mmdagg'])
ap.add_argument('--unit', required=True)
ap.add_argument('--feats_dir', required=True)
ap.add_argument('--only', required=True, help='comma list of epochs')
ap.add_argument('--out_dir', required=True)
a = ap.parse_args()

eps = sorted(set(int(x) for x in a.only.split(',')))
eps = [e for e in eps
       if os.path.exists(os.path.join(a.feats_dir, 'ep%03d.npz' % e))]
os.makedirs(a.out_dir, exist_ok=True)
out = os.path.join(a.out_dir, 'fill2_%s_%s.csv' % (a.method, a.unit))
f = open(out, 'w')
f.write('model,epoch,budget,stat,seed,acc,p\n')
print('task %s %s epochs %s' % (a.method, a.unit, eps), flush=True)

for ep in eps:
    z = np.load(os.path.join(a.feats_dir, 'ep%03d.npz' % ep))
    e_tr = l2n(z['h_tr'].astype(np.float64))
    e_te = l2n(z['h_te'].astype(np.float64))
    if a.method == 'c2st':
        for seed in range(42, 47):
            acc, p = c2st_test(e_tr, e_te, n_sub=10000, seed=seed)
            f.write('%s,%d,b10k,c2st,%d,%.6f,%.6f\n' % (a.unit, ep, seed, acc, p))
    elif a.method == 'ed':
        for seed in range(42, 47):
            _, p = ed_cached(e_tr, e_te, 10000, 200, seed=seed)
            f.write('%s,%d,b10k,ed,%d,nan,%.6f\n' % (a.unit, ep, seed, p))
    else:
        for seed in (42, 43):
            try:
                rej = mmdagg_test(e_tr, e_te, n_sub=5000, seed=seed)
                f.write('%s,%d,b5000,mmdagg,%d,nan,%.6f\n' % (a.unit, ep, seed, float(rej)))
            except Exception as exc:
                print('mmdagg fail ep%d: %s' % (ep, exc), flush=True)
    f.flush()
    print('ep%d done' % ep, flush=True)
f.close()
print('->', out)
