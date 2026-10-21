"""gpufill.py -- GPU fill of non-grid epochs for ED(b10k) + MMD(b10k, both
spaces). Exact port: numpy RNG consumed in the original order; matrices in
float64 (TF32 off), stored float32 like the originals. Self-check on the
first epoch compares against the CPU originals and aborts on mismatch."""
import argparse, os, re
import numpy as np, torch
from baselines2_stats import l2n, ed_cached
from unnorm_stats import mmd_fast

torch.backends.cuda.matmul.allow_tf32 = False
dev = 'cuda'

def ed_gpu(emb_tr, emb_te, n_sub, n_perm, seed):
    rng = np.random.RandomState(seed)
    X = emb_tr[rng.permutation(len(emb_tr))[:n_sub]]
    Y = emb_te[rng.permutation(len(emb_te))[:n_sub]]
    pooled = torch.tensor(np.concatenate([X, Y]), dtype=torch.float64, device=dev)
    n, m = len(X), len(Y); N = n + m
    sq = (pooled**2).sum(1, keepdim=True)
    d2 = torch.clamp(sq + sq.T - 2 * pooled @ pooled.T, min=0)
    Dm = d2.sqrt_().to(torch.float32); del d2
    Dm.fill_diagonal_(0)
    tot = float(Dm.sum())
    def stat(idx):
        z = torch.zeros(N, dtype=torch.float32, device=dev); z[idx] = 1
        S = Dm @ z
        xx = float(z @ S); sx = float(S.sum()); xy = sx - xx
        yy = tot - xx - 2 * xy
        return 2*xy/(n*m) - xx/(n*n) - yy/(m*m)
    obs = stat(torch.arange(n, device=dev))
    count = 0
    for _ in range(n_perm):
        idx = torch.tensor(rng.permutation(N)[:n].copy(), device=dev)
        if stat(idx) >= obs: count += 1
    return obs, (count + 1) / (n_perm + 1)

def mmd_gpu(X, Y, n_perm, seed):
    rng = np.random.RandomState(seed)
    pooled_np = np.concatenate([X, Y]).astype(np.float64)
    n, m = len(X), len(Y); N = n + m
    sub = pooled_np[rng.choice(N, min(2000, N), replace=False)]
    d2s = np.sum(sub**2, 1, keepdims=True) + np.sum(sub**2, 1, keepdims=True).T - 2*sub@sub.T
    d2s = np.maximum(d2s, 0.0); np.fill_diagonal(d2s, np.inf)
    vals = d2s[np.isfinite(d2s) & (d2s > 0)]
    sigma = max(float(np.sqrt(np.median(vals))), 1e-8) if len(vals) else 1.0
    pooled = torch.tensor(pooled_np, dtype=torch.float64, device=dev)
    sq = (pooled**2).sum(1, keepdim=True)
    d2 = torch.clamp(sq + sq.T - 2 * pooled @ pooled.T, min=0)
    Km = torch.exp(-d2 / (2 * sigma**2)).to(torch.float32); del d2
    Km.fill_diagonal_(0)
    tot = float(Km.sum())
    def stat(idx):
        z = torch.zeros(N, dtype=torch.float32, device=dev); z[idx] = 1
        S = Km @ z
        Kxx = float(z @ S); sx = float(S.sum())
        Kyy = tot - 2*sx + Kxx; Kxy = sx - Kxx
        return Kxx/(n*(n-1)) + Kyy/(m*(m-1)) - 2*Kxy/(n*m)
    obs = stat(torch.arange(n, device=dev))
    count = 0
    for _ in range(n_perm):
        idx = torch.tensor(rng.permutation(N)[:n].copy(), device=dev)
        if stat(idx) >= obs: count += 1
    return obs, (count + 1) / (n_perm + 1)

ap = argparse.ArgumentParser()
ap.add_argument('--unit', required=True)
ap.add_argument('--feats_dir', required=True)
ap.add_argument('--epochs', required=True)
ap.add_argument('--out_dir', required=True)
a = ap.parse_args()
ONLY = sorted(set(int(x) for x in a.epochs.split(',')))
eps = sorted(int(f[2:-4]) for f in os.listdir(a.feats_dir)
             if re.fullmatch(r'ep\d+\.npz', f))
eps = [e for e in eps if e in ONLY]
os.makedirs(a.out_dir, exist_ok=True)
fed = open(os.path.join(a.out_dir, f'fill2_ed_{a.unit}.csv.discard'), 'w')
fmm = open(os.path.join(a.out_dir, f'fill2_mmd_{a.unit}.csv'), 'w')
fed.write('model,epoch,budget,stat,seed,acc,p\n')
fmm.write('model,epoch,space,budget,stat,seed,T,p\n')

first = True
for ep in eps:
    z = np.load(os.path.join(a.feats_dir, f'ep{ep:03d}.npz'))
    raw = z['h_tr'].astype(np.float64), z['h_te'].astype(np.float64)
    e_tr, e_te = l2n(raw[0]), l2n(raw[1])
    if first:   # self-check against CPU originals, seed 42
        _, p_cpu = ed_cached(e_tr, e_te, 10000, 200, seed=42)
        _, p_gpu = ed_gpu(e_tr, e_te, 10000, 200, seed=42)
        print(f'SELFCHECK ed ep{ep}: cpu {p_cpu:.6f} gpu {p_gpu:.6f}', flush=True)
        assert abs(p_cpu - p_gpu) <= 0.05, 'ED self-check failed'
        rng = np.random.RandomState(42)
        Xc = e_tr[rng.permutation(len(e_tr))[:10000]]
        Yc = e_te[rng.permutation(len(e_te))[:10000]]
        _, q_cpu = mmd_fast(Xc, Yc, 200, seed=42)
        _, q_gpu = mmd_gpu(Xc, Yc, 200, seed=42)
        print(f'SELFCHECK mmd ep{ep}: cpu {q_cpu:.6f} gpu {q_gpu:.6f}', flush=True)
        assert abs(q_cpu - q_gpu) <= 0.05, 'MMD self-check failed'
        first = False
    for seed in range(42, 47):
        _, p = ed_gpu(e_tr, e_te, 10000, 200, seed=seed)
        fed.write(f'{a.unit},{ep},b10k,ed,{seed},nan,{p:.6f}\n')
    for sp, (atr, ate) in {'l2': (e_tr, e_te), 'unnorm': raw}.items():
        for seed in range(42, 47):
            rng = np.random.RandomState(seed)
            X = atr[rng.permutation(len(atr))[:10000]]
            Y = ate[rng.permutation(len(ate))[:10000]]
            _, p = mmd_gpu(X, Y, 200, seed=seed)
            fmm.write(f'{a.unit},{ep},{sp},b10k,mmd,{seed},nan,{p:.6f}\n')
    fed.flush(); fmm.flush()
    print(f'ep{ep} done', flush=True)
fed.close(); fmm.close()
print('done', flush=True)
