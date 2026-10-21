"""LEGACY (camera-ready code release note): small-budget VRT runner of the rebuttal (k = 15, seeds 42-51); the current entry point is vrt_forward_budget.py --budget small with vrt_low_lib.vrt_low (verbatim copy of this scoring). Kept as run for provenance; not an entry point of the release."""
"""ksweep15_fill.py - small-budget VRT (k=15) on the full epoch grid.
Protocol: M=500 train reference + 1,000 train queries + 1,500 test queries,
k=15, 2,000 permutations, seeds 42-51, two-sided median test. Scoring
copied verbatim from the supplementary phase_multiseed_vrt.py.
Output: ksweep15_{unit}_{lo}-{hi}.csv, columns model,epoch,k,stat,seed,T,p."""
import argparse, csv, glob, os, re
import numpy as np
from scipy.special import betainc, betaln

def log_cap_volume(theta, D):
    sin2 = np.sin(theta)**2
    I = betainc((D-1)/2.0, 0.5, sin2)
    V = np.where(theta <= np.pi/2, 0.5*I, 1.0 - 0.5*I)
    safe = V > 1e-300
    log_V = np.empty_like(V)
    log_V[safe] = np.log(V[safe])
    if np.any(~safe):
        a = (D-1) / 2.0
        const = -np.log(2.0) - np.log(a) - betaln(a, 0.5)
        log_V[~safe] = const + (D-1) * np.log(np.maximum(np.sin(theta[~safe]), 1e-300))
    return log_V

def compute_chat(query, ref, D, k, batch_size=1024):
    Nq = query.shape[0]; theta_all = []
    for s in range(0, Nq, batch_size):
        e = min(s+batch_size, Nq)
        cos = query[s:e] @ ref.T
        pidx = np.argpartition(-cos, k - 1, axis=1)[:, :k]
        top = np.take_along_axis(cos, pidx, axis=1)
        top = np.take_along_axis(top, np.argsort(-top, axis=1), axis=1)
        theta_all.append(np.arccos(np.clip(top, -1, 1)))
    theta = np.concatenate(theta_all, 0)
    L = log_cap_volume(theta, D); dL = np.diff(L, axis=1)
    r = np.arange(1, k, dtype=np.float64)
    Z = np.maximum(r[None,:] * dL, 0.0)
    return np.maximum(Z.mean(axis=1), 1e-12)

def vrt_low(e_tr, e_te, D, k=15, M=500, n_qtr=1000, n_qte=1500,
            n_perm=2000, seed=42):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(e_tr))
    ref = e_tr[perm[:M]]
    q_tr = e_tr[perm[M:M+min(n_qtr, len(e_tr)-M)]]
    pte = rng.permutation(len(e_te))
    q_te = e_te[pte[:min(n_qte, len(e_te))]]
    c_tr = compute_chat(q_tr, ref, D, k)
    c_te = compute_chat(q_te, ref, D, k)
    obs = np.median(c_te) - np.median(c_tr)
    pooled = np.concatenate([c_tr, c_te]); na = len(c_tr)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled))
        if abs(np.median(pooled[p[na:]]) - np.median(pooled[p[:na]])) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

ap = argparse.ArgumentParser()
ap.add_argument('--unit', required=True)
ap.add_argument('--feats_dir', required=True)
ap.add_argument('--out_dir', required=True)
ap.add_argument('--lo', type=int, default=0)
ap.add_argument('--hi', type=int, default=9999)
a = ap.parse_args()

out = os.path.join(a.out_dir, f'ksweep15_{a.unit}_{a.lo}-{a.hi}.csv')
done = set()
if os.path.exists(out):
    with open(out) as fh:
        for r in csv.DictReader(fh):
            try: done.add((int(r['epoch']), int(r['seed'])))
            except (KeyError, ValueError): pass
    f = open(out, 'a', newline='')
    w = csv.writer(f)
else:
    f = open(out, 'w', newline='')
    w = csv.writer(f)
    w.writerow(['model', 'epoch', 'k', 'stat', 'seed', 'T', 'p'])

files = sorted(glob.glob(os.path.join(a.feats_dir, 'ep*.npz')))
files = [fp for fp in files
         if a.lo <= int(re.search(r'ep(\d+)\.npz', os.path.basename(fp)).group(1)) <= a.hi]
print(f'{a.unit} [{a.lo},{a.hi}]: {len(files)} checkpoints, {len(done)} done', flush=True)
for fp in files:
    ep = int(re.search(r'ep(\d+)\.npz', os.path.basename(fp)).group(1))
    z = None
    for seed in range(42, 52):
        if (ep, seed) in done: continue
        if z is None:
            z = np.load(fp)
            e_tr = l2n(z['h_tr'].astype(np.float64))
            e_te = l2n(z['h_te'].astype(np.float64))
            D = e_tr.shape[1]
        T, p = vrt_low(e_tr, e_te, D, seed=seed)
        w.writerow([a.unit, ep, 15, 'chat', seed, f'{T:.6e}', f'{p:.6f}'])
    f.flush()
    if z is not None:
        print(f'ep{ep:03d} done', flush=True)
f.close()
print('->', out, flush=True)
