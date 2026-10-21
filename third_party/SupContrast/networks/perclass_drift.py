"""
perclass_drift.py — VRT-independent evidence of backbone divergence
===================================================================
Extends the paper's Appendix J per-class decomposition (Figure 10, ResNet-50
only) to any model with extracted features. Answers nxR4 Q5: the per-class
train/test centroid drift theta_c and concentration gap dR_c are computed
from class means alone — no kNN, no cap volumes, no permutation test — so
they certify divergence independently of VRT, and simultaneously explain
why global tests miss it (directional cancellation across classes).

For each epoch npz:
  theta_bar  : mean over classes of angle between train/test class centroids
  theta_glob : same angle computed on the global (all-class) centroids
  dR_bar     : mean over classes of (R_tr - R_te), R = ||class mean of
               l2-normalized embeddings|| (concentration; 1 = collapsed)
  dR_glob    : global counterpart

Usage (needs the dataset for labels; extraction order is canonical):
  python perclass_drift.py --model v7   --dataset cifar10  --feats_dir <...>/v7
  python perclass_drift.py --model ablA --dataset cifar10  --feats_dir <...>/ablA
  python perclass_drift.py --model r50  --dataset cifar100 --feats_dir <...>/r50
Output: perclass_{model}.csv
"""
import argparse
import os
import re

import numpy as np
import torchvision

DATA_FOLDER = '/home3/p318770/venvs/overfitting'


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def centroid_stats(e_tr, e_te, y_tr, y_te, n_classes):
    thetas, dRs = [], []
    for c in range(n_classes):
        m_tr = e_tr[y_tr == c].mean(axis=0)
        m_te = e_te[y_te == c].mean(axis=0)
        cos = np.dot(m_tr, m_te) / (np.linalg.norm(m_tr) * np.linalg.norm(m_te))
        thetas.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        dRs.append(np.linalg.norm(m_tr) - np.linalg.norm(m_te))
    g_tr = e_tr.mean(axis=0)
    g_te = e_te.mean(axis=0)
    cosg = np.dot(g_tr, g_te) / (np.linalg.norm(g_tr) * np.linalg.norm(g_te))
    theta_glob = np.degrees(np.arccos(np.clip(cosg, -1, 1)))
    dR_glob = np.linalg.norm(g_tr) - np.linalg.norm(g_te)
    return float(np.mean(thetas)), theta_glob, float(np.mean(dRs)), float(dR_glob)


def split_half_null(e_tr, y_tr, y_te, n_classes, n_rep=5, seed=0):
    """Noise floor: per-class centroid angle between two DISJOINT train
    subsets, the pseudo-test one matched in per-class size to the real test
    set. Same distribution by construction -> any angle is pure sampling
    noise at exactly the train-vs-test noise level."""
    rng = np.random.RandomState(seed)
    te_counts = np.bincount(y_te, minlength=n_classes)
    reps = []
    for _ in range(n_rep):
        thetas = []
        for c in range(n_classes):
            idx = np.where(y_tr == c)[0]
            idx = idx[rng.permutation(len(idx))]
            k = te_counts[c]
            m_a = e_tr[idx[:k]].mean(axis=0)          # pseudo-test half
            m_b = e_tr[idx[k:]].mean(axis=0)          # remaining train
            cos = np.dot(m_a, m_b) / (np.linalg.norm(m_a) * np.linalg.norm(m_b))
            thetas.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        reps.append(np.mean(thetas))
    return float(np.mean(reps)), float(np.std(reps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--dataset', choices=['cifar10', 'cifar100'], required=True)
    ap.add_argument('--feats_dir', required=True)
    ap.add_argument('--out_dir', default='.')
    args = ap.parse_args()

    cls = (torchvision.datasets.CIFAR10 if args.dataset == 'cifar10'
           else torchvision.datasets.CIFAR100)
    y_tr = np.array(cls(DATA_FOLDER, True, download=False).targets)
    y_te = np.array(cls(DATA_FOLDER, False, download=False).targets)
    n_classes = 10 if args.dataset == 'cifar10' else 100

    files = sorted(f for f in os.listdir(args.feats_dir)
                   if re.fullmatch(r'ep\d+\.npz', f))
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f'perclass_{args.model}.csv')
    with open(out_csv, 'w') as f:
        f.write('model,epoch,theta_bar_deg,theta_glob_deg,'
                'theta_null_deg,theta_null_std,dR_bar,dR_glob\n')
        for name in files:
            ep = int(name[2:-4])
            z = np.load(os.path.join(args.feats_dir, name))
            e_tr = l2n(z['h_tr'].astype(np.float64))
            e_te = l2n(z['h_te'].astype(np.float64))
            tb, tg, rb, rg = centroid_stats(e_tr, e_te, y_tr, y_te, n_classes)
            tn, tns = split_half_null(e_tr, y_tr, y_te, n_classes, seed=ep)
            f.write(f'{args.model},{ep},{tb:.4f},{tg:.4f},{tn:.4f},{tns:.4f},'
                    f'{rb:+.6f},{rg:+.6f}\n')
            f.flush()
            print(f'ep{ep:03d}: train-vs-test {tb:.2f} deg | '
                  f'noise floor {tn:.2f}±{tns:.2f} deg | global {tg:.2f} deg',
                  flush=True)
    print('->', out_csv)


if __name__ == '__main__':
    main()
