"""Per-epoch norm characterization for the SSL runs (Q1 table SSL rows).
d_med = 100*(med||te||/med||tr|| - 1); d_disp = 100*(MAD_te/MAD_tr - 1).
Reads cached fp16 features; one npz at a time."""
import csv, glob, os
import numpy as np
ROOT = '/scratch/p318770/model_zoos/ssl_supcon_feats'
out_rows = []
for d in sorted(glob.glob(os.path.join(ROOT, 'feats_*'))):
    tag = os.path.basename(d)
    for f in sorted(glob.glob(os.path.join(d, 'h_ep*.npz'))):
        ep = int(os.path.basename(f)[4:-4])
        z = np.load(f)
        ntr = np.linalg.norm(z['h_tr'].astype(np.float32), axis=1)
        nte = np.linalg.norm(z['h_te'].astype(np.float32), axis=1)
        mad = lambda v: np.median(np.abs(v - np.median(v)))
        d_med = 100 * (np.median(nte) / np.median(ntr) - 1)
        d_disp = 100 * (mad(nte) / mad(ntr) - 1)
        out_rows.append((tag, ep, np.median(ntr), np.median(nte), d_med,
                         mad(ntr), mad(nte), d_disp))
        print(f'{tag} ep{ep:4d}  med_tr {np.median(ntr):.4f}  med_te '
              f'{np.median(nte):.4f}  d_med {d_med:+.2f}%  d_disp {d_disp:+.2f}%',
              flush=True)
with open('/scratch/p318770/model_zoos/ladder_out/ssl_norm_stats.csv', 'w',
          newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['run', 'epoch', 'med_tr', 'med_te', 'd_med_pct',
                'mad_tr', 'mad_te', 'd_disp_pct'])
    w.writerows(out_rows)
print('done')
