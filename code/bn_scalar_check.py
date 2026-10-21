"""
bn_scalar_check.py - is the eval<->batch BN switch a per-split scalar on embeddings?
====================================================================================
Uses ONLY existing feature files (no GPU, no re-extraction): every pair
{X, X_bntrain} under the feats root, same checkpoint, same samples, same order.

Scalar-chain hypothesis: h_eval ~= c * h_batch, one positive factor per split.
  P1 per-sample cos(h_eval, h_batch) ~= 1          (directions intact)
  P2 per-sample ratio ||h_eval||/||h_batch|| ~= const per split
  P3 c_te/c_tr != 1 = the spurious train-test length gap injected into the
     raw-norm channel by the extraction protocol
General affine alternative (per-channel D, shift b): cos visibly < 1, ratios
per-sample dependent, no exact cancellation guarantee for L2 normalization.
"""
import argparse, csv, glob, os, re
import numpy as np

CAND = [('h_tr','h_te'), ('train','test'), ('tr','te'),
        ('X_tr','X_te'), ('h_train','h_test')]

def load_pair_file(path):
    f = np.load(path)
    keys = list(f.files)
    for ktr, kte in CAND:
        if ktr in keys and kte in keys:
            return f[ktr], f[kte], (ktr, kte)
    if len(keys) >= 2:
        return f[keys[0]], f[keys[1]], tuple(keys[:2])
    raise ValueError(f'{path}: unusable keys {keys}')

def stats(he, hb):
    he = he.astype(np.float64); hb = hb.astype(np.float64)
    ne = np.linalg.norm(he, axis=1); nb = np.linalg.norm(hb, axis=1)
    ok = (ne > 0) & (nb > 0)
    he, hb, ne, nb = he[ok], hb[ok], ne[ok], nb[ok]
    ratio = ne / nb
    cos = np.clip(np.einsum('ij,ij->i', he, hb) / (ne * nb), -1.0, 1.0)
    mr = float(np.median(ratio))
    return dict(med_ratio=mr,
                mad_rel=float(np.median(np.abs(ratio - mr)) / mr),
                med_cos=float(np.median(cos)),
                p05_cos=float(np.quantile(cos, 0.05)),
                med_ang=float(np.median(np.degrees(np.arccos(cos)))),
                n=int(len(ratio)))

def eps_of(d):
    out = {}
    for f in glob.glob(os.path.join(d, 'ep*.npz')):
        m = re.match(r'ep(\d+)\.npz$', os.path.basename(f))
        if m: out[int(m.group(1))] = f
    return out

def pick(eps, every, keep):
    eps = sorted(eps)
    if not eps: return []
    sel = set(e for e in eps if e % every == 0) | {eps[0], eps[-1]}
    sel |= (set(keep) & set(eps))
    return sorted(sel)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', default='/scratch/p318770/model_zoos/ladder_feats')
    ap.add_argument('--every', type=int, default=10)
    ap.add_argument('--keep', type=int, nargs='*', default=[44, 48, 61, 76])
    ap.add_argument('--pairs', nargs='*', default=None,
                    help='base names to check; default = all *_bntrain found')
    ap.add_argument('--out', default='/scratch/p318770/model_zoos/ladder_out/bn_scalar_check.csv')
    a = ap.parse_args()

    bases = a.pairs
    if bases is None:
        bases = sorted(os.path.basename(d)[:-len('_bntrain')]
                       for d in glob.glob(os.path.join(a.feats, '*_bntrain'))
                       if os.path.isdir(d))
    fh = open(a.out, 'w', newline=''); w = csv.writer(fh)
    w.writerow(['model','epoch','split','med_ratio','mad_rel','med_cos',
                'p05_cos','med_ang_deg','n'])
    for base in bases:
        d_ev = os.path.join(a.feats, base)
        d_bt = os.path.join(a.feats, base + '_bntrain')
        if not os.path.isdir(d_ev):
            print(f'{base}: eval dir missing ({d_ev}), skip'); continue
        ev, bt = eps_of(d_ev), eps_of(d_bt)
        common = sorted(set(ev) & set(bt))
        sel = pick(common, a.every, a.keep)
        print(f'\n#### {base}: {len(common)} common epochs, checking {sel}', flush=True)
        shown = False
        for ep in sel:
            e_tr, e_te, k1 = load_pair_file(ev[ep])
            b_tr, b_te, k2 = load_pair_file(bt[ep])
            if not shown:
                print(f'  keys eval={k1} bntrain={k2} shapes {e_tr.shape}/{e_te.shape}')
                shown = True
            if e_tr.shape != b_tr.shape or e_te.shape != b_te.shape:
                print(f'  ep{ep:03d} SHAPE MISMATCH, skip'); continue
            s_tr = stats(e_tr, b_tr); s_te = stats(e_te, b_te)
            for split, s in (('tr', s_tr), ('te', s_te)):
                w.writerow([base, ep, split, f"{s['med_ratio']:.5f}",
                            f"{s['mad_rel']:.5f}", f"{s['med_cos']:.6f}",
                            f"{s['p05_cos']:.6f}", f"{s['med_ang']:.3f}", s['n']])
                warn = '  ORDER-MISMATCH?' if s['med_cos'] < 0.5 else ''
                print(f"  ep{ep:03d} {split}  ratio {s['med_ratio']:.4f} "
                      f"(relMAD {s['mad_rel']:.4f})  cos med {s['med_cos']:.6f} "
                      f"p05 {s['p05_cos']:.6f}  ang {s['med_ang']:.3f}deg{warn}",
                      flush=True)
            r = s_te['med_ratio'] / s_tr['med_ratio']
            w.writerow([base, ep, 'te_over_tr', f'{r:.5f}', '', '', '', '', ''])
            print(f"  ep{ep:03d} c_te/c_tr {r:.5f}", flush=True)
        fh.flush()
    fh.close()
    print(f'\nCSV -> {a.out}', flush=True)

main()
