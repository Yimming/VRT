#!/usr/bin/env python3
"""
ssl_linear_probe.py — linear-probe anchor for the SSL "benign vs structural"
claim (check-report item C10).

For each selected epoch: load stored SimCLR features (train & test split),
fit a multinomial logistic probe on the TRAIN features, report train/test
accuracy and NLL. The deliverable is the trajectory of the train-test gap.

Pre-registered read-out (decided before looking at results):
  * If the probe gap (acc and NLL) stays flat through the norm-drift onset
    (~ep45 cifar10 / ~ep110 cifar100) and opens around the VRT onset
    (~140 / ~200), the "benign scale drift vs structural drift" wording gets
    an independent label-based anchor — the SSL analogue of F2's NLL anchor.
  * Any other outcome: keep the numbers, and downgrade "benignly" in the
    draft to "without accompanying structural change detectable by VRT".

ASSUMPTIONS — VERIFY BEFORE TRUSTING OUTPUT:
  [A1] Feature files are in original dataset order (extraction DataLoader
       ran with shuffle=False), OR the npz contains labels under one of:
       labels / label / y / targets. With order-based labels the script
       prints a sanity line per epoch: late-epoch TEST accuracy must be far
       above chance (cifar10 >> 10%, cifar100 >> 1%); near-chance means
       misalignment and the results are garbage.
  [A2] If the npz stores both encoder features and projection-head features,
       probe the ENCODER features (standard SSL protocol) — select with
       --key. Projection features also work for a gap trajectory but give
       lower absolute accuracy; state which one was used.

Usage (adapt the templates to your layout; {epoch} is substituted, format
specs like {epoch:04d} allowed):

  python3 ssl_linear_probe.py \
      --train-tpl '/scratch/p318770/.../cifar10/feats_train_ep{epoch}.npz' \
      --test-tpl  '/scratch/p318770/.../cifar10/feats_test_ep{epoch}.npz' \
      --dataset cifar10 --data-root /path/used/in/training \
      --epochs 1,20,45,100,140,200,300,500,750,1000 \
      --out probe_cifar10.csv

Notes:
  * If one npz per epoch holds BOTH splits, give only --train-tpl; the
    script looks for train*/test* keys and lists what it found on failure.
  * fp16 features are cast to float32.
  * --subsample 20000 for a quick pass (fit on subsample, evaluate on full).
  * Runtime ~2-10 min per epoch point at 50k x 2048; submit via sbatch on a
    CPU partition, not on the login node.

  cat > job_probe.sh <<'EOF'
  #!/bin/bash
  #SBATCH --job-name=sslprobe
  #SBATCH --partition=regular
  #SBATCH --cpus-per-task=8
  #SBATCH --time=04:00:00
  #SBATCH --mem=32G
  #SBATCH --output=sslprobe_%j.out
  python3 ssl_linear_probe.py --train-tpl '...' --test-tpl '...' \
      --dataset cifar10 --data-root ... --epochs 1,20,45,100,140,200,300,500,750,1000
  EOF
"""

import argparse
import csv
import sys

import numpy as np

LABEL_KEYS = ('labels', 'label', 'y', 'targets', 't')


def _pick_feature_array(d, key=None, want=None):
    """Pick a 2-D float array from an npz. `want` filters key names."""
    if key is not None:
        return np.asarray(d[key]), key
    cands = []
    for k in d.files:
        a = d[k]
        if getattr(a, 'ndim', 0) == 2 and a.dtype.kind == 'f':
            if want is None or any(w in k.lower() for w in want):
                cands.append((a.shape[0] * a.shape[1], k))
    if not cands:
        raise KeyError(f'no 2-D float array found (keys: {list(d.files)}); '
                       f'pass --key explicitly')
    cands.sort(reverse=True)
    k = cands[0][1]
    return np.asarray(d[k]), k


def _pick_labels(d):
    for k in d.files:
        if k.lower() in LABEL_KEYS:
            a = np.asarray(d[k]).ravel()
            if a.dtype.kind in 'iu' or np.allclose(a, a.astype(int)):
                return a.astype(np.int64)
    return None


def load_epoch(train_tpl, test_tpl, epoch, key):
    if test_tpl:
        dtr = np.load(train_tpl.format(epoch=epoch), allow_pickle=True)
        dte = np.load(test_tpl.format(epoch=epoch), allow_pickle=True)
        Xtr, ktr = _pick_feature_array(dtr, key)
        Xte, kte = _pick_feature_array(dte, key)
        ytr, yte = _pick_labels(dtr), _pick_labels(dte)
    else:  # combined file: look for train*/test* keys
        d = np.load(train_tpl.format(epoch=epoch), allow_pickle=True)
        Xtr, ktr = _pick_feature_array(d, key, want=('train', 'tr'))
        Xte, kte = _pick_feature_array(d, None, want=('test', 'te', 'val'))
        ytr = yte = None
        for k in d.files:
            kl = k.lower()
            if any(kl == f'{p}_{s}' or kl == f'{s}_{p}' for p in LABEL_KEYS
                   for s in ('train', 'tr')):
                ytr = np.asarray(d[k]).ravel().astype(np.int64)
            if any(kl == f'{p}_{s}' or kl == f'{s}_{p}' for p in LABEL_KEYS
                   for s in ('test', 'te', 'val')):
                yte = np.asarray(d[k]).ravel().astype(np.int64)
    return (Xtr.astype(np.float32), Xte.astype(np.float32),
            ytr, yte, ktr, kte)


def dataset_labels(name, root):
    """[A1] order-based labels; requires extraction with shuffle=False."""
    from torchvision import datasets
    cls = {'cifar10': datasets.CIFAR10, 'cifar100': datasets.CIFAR100}[name]
    tr = cls(root, train=True, download=False)
    te = cls(root, train=False, download=False)
    return np.asarray(tr.targets, np.int64), np.asarray(te.targets, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-tpl', required=True)
    ap.add_argument('--test-tpl', default=None)
    ap.add_argument('--dataset', choices=['cifar10', 'cifar100'], required=True)
    ap.add_argument('--data-root', default='./data',
                    help='torchvision root used during training (for labels '
                         'when the npz has none)')
    ap.add_argument('--epochs', required=True,
                    help='comma-separated, e.g. 1,20,45,100,140,200,300,500,750,1000')
    ap.add_argument('--key', default=None,
                    help='[A2] feature key in the npz (prefer encoder feats)')
    ap.add_argument('--l2norm', action='store_true',
                    help='l2-normalize features instead of standardizing')
    ap.add_argument('--subsample', type=int, default=0,
                    help='fit on N random train rows (eval on full splits)')
    ap.add_argument('--C', type=float, default=1.0)
    ap.add_argument('--max-iter', type=int, default=500)
    ap.add_argument('--out', default='probe_results.csv')
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    epochs = [int(e) for e in args.epochs.split(',')]
    chance = 0.1 if args.dataset == 'cifar10' else 0.01
    fallback = None
    rows = []

    for ep in epochs:
        try:
            Xtr, Xte, ytr, yte, ktr, kte = load_epoch(
                args.train_tpl, args.test_tpl, ep, args.key)
        except (FileNotFoundError, KeyError) as e:
            print(f'[ep {ep}] SKIP: {e}', flush=True)
            continue

        if ytr is None or yte is None:
            if fallback is None:
                fallback = dataset_labels(args.dataset, args.data_root)
                print('[A1] npz has no labels -> using dataset-order labels; '
                      'extraction must have used shuffle=False', flush=True)
            ytr, yte = fallback
        if len(ytr) != len(Xtr) or len(yte) != len(Xte):
            print(f'[ep {ep}] FATAL: feature/label length mismatch '
                  f'({len(Xtr)}/{len(ytr)} train, {len(Xte)}/{len(yte)} test)')
            sys.exit(1)

        if args.l2norm:
            Xtr = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True)
            Xte = Xte / np.linalg.norm(Xte, axis=1, keepdims=True)
        else:
            mu = Xtr.mean(0, keepdims=True)
            sd = Xtr.std(0, keepdims=True) + 1e-8
            Xtr = (Xtr - mu) / sd
            Xte = (Xte - mu) / sd

        if args.subsample and args.subsample < len(Xtr):
            idx = np.random.RandomState(0).choice(len(Xtr), args.subsample,
                                                  replace=False)
            Xfit, yfit = Xtr[idx], ytr[idx]
        else:
            Xfit, yfit = Xtr, ytr

        clf = LogisticRegression(C=args.C, solver='lbfgs',
                                 max_iter=args.max_iter, n_jobs=-1)
        clf.fit(Xfit, yfit)

        classes = clf.classes_
        Ptr, Pte = clf.predict_proba(Xtr), clf.predict_proba(Xte)
        acc_tr = float((classes[Ptr.argmax(1)] == ytr).mean())
        acc_te = float((classes[Pte.argmax(1)] == yte).mean())
        nll_tr = float(log_loss(ytr, Ptr, labels=classes))
        nll_te = float(log_loss(yte, Pte, labels=classes))

        flag = ' <-- near chance: check [A1] alignment!' \
            if acc_te < 3 * chance else ''
        print(f'[ep {ep:4d}] key={ktr}/{kte} dim={Xtr.shape[1]} | '
              f'acc {acc_tr:.4f}/{acc_te:.4f} gap {acc_tr - acc_te:+.4f} | '
              f'nll {nll_tr:.4f}/{nll_te:.4f} gap {nll_te - nll_tr:+.4f}{flag}',
              flush=True)
        rows.append((ep, len(Xfit), acc_tr, acc_te, acc_tr - acc_te,
                     nll_tr, nll_te, nll_te - nll_tr))

    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'n_fit', 'acc_train', 'acc_test', 'acc_gap',
                    'nll_train', 'nll_test', 'nll_gap'])
        w.writerows(rows)

    print('\n| epoch | acc_train | acc_test | acc_gap | nll_train | nll_test | nll_gap |')
    print('|---|---|---|---|---|---|---|')
    for r in rows:
        print(f'| {r[0]} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:+.4f} '
              f'| {r[5]:.4f} | {r[6]:.4f} | {r[7]:+.4f} |')
    print('\nDone. CSV:', args.out)


if __name__ == '__main__':
    main()
