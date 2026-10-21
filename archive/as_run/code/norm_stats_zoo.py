"""Norm characterization (d_med/d_disp) for zoo m1-m4 — last gap in the
nxR4-Q1 table (m0 was done as representative)."""
import glob, os
import numpy as np
F = '/scratch/p318770/model_zoos/ladder_feats'
mad = lambda v: np.median(np.abs(v - np.median(v)))
for zoo in ('zoo_c10', 'zoo_c100'):
    for mi in (1, 2, 3, 4):
        d = f'{F}/{zoo}_m{mi}'
        files = sorted(glob.glob(os.path.join(d, 'ep*.npz')))
        if not files:
            print(f'{zoo}_m{mi}: no feats'); continue
        for tag, f in (('early', files[0]), ('final', files[-1])):
            z = np.load(f)
            ntr = np.linalg.norm(z['h_tr'].astype(np.float32), axis=1)
            nte = np.linalg.norm(z['h_te'].astype(np.float32), axis=1)
            print(f'{zoo}_m{mi} {tag} ep{int(os.path.basename(f)[2:-4]):3d}  '
                  f'd_med {100*(np.median(nte)/np.median(ntr)-1):+.2f}%  '
                  f'd_disp {100*(mad(nte)/mad(ntr)-1):+.2f}%', flush=True)
