"""
imagenet_randinit_extract.py — the random-initialization ImageNet null of Appendix C, as feature files, so that
every two-sample test can be run on it with the standard scripts.

For init i = 0..N-1: torch.manual_seed(seed0 + i), a fresh DeepResNet50 with random weights (default PyTorch
initialization, no checkpoint), features of the SAME fixed 50,000 training and 10,000 validation images as all
other ImageNet runs (data seeds 42 / 123, verified against the eval-mode checkpoint files), saved to
    /scratch/p318770/model_zoos/ladder_feats/imgnet_rand_<bn_mode>/ep{i:03d}.npz      (h_tr, h_te, idx_tr, idx_te)
The init index plays the role of the epoch, so  --epochs 0-29  in the test scripts runs one test per init.
With --pair trtr the two samples come from one distribution, so the rejection rate over inits (one analysis seed per
init) is the Type-I rate of a method. With --pair trval the backbone is untrained but the two ImageNet splits may
still differ at the input level, so rejections there measure that split difference, not a Type-I error.

--bn_mode eval  : BatchNorm uses its initial running statistics (mean 0, var 1), i.e. no normalization at all
--bn_mode train : per-batch statistics, the extraction of the Section 4.4 control
--pair trval    : the 50,000 train vs 10,000 validation images of every other ImageNet run (two different ImageNet splits)
--pair trtr     : the same 50,000 train images vs 10,000 further train images (next in the same shuffle), two disjoint
                  draws from one distribution: the null hypothesis of a two-sample test holds by construction
                  -> ladder_feats/imgnet_rand_<bn_mode>_trtr/ep{i:03d}.npz
Requires the full ImageNet under $TMPDIR/ILSVRC/Data/CLS-LOC (see job_imgnet_randinit.sh).
"""
import os, sys, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imagenet_extract import DeepResNet50, make_loaders, extract_raw, DEVICE, OUT_ROOT, N_TRAIN_FULL, N_VAL_FULL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_init', type=int, default=30)
    ap.add_argument('--seed0', type=int, default=1000, help='init i uses torch.manual_seed(seed0 + i)')
    ap.add_argument('--bn_mode', choices=['eval', 'train'], default='eval')
    ap.add_argument('--pair', choices=['trval', 'trtr'], default='trval',
                    help='trval: the 50,000 train vs 10,000 validation images of every other ImageNet run; '
                         'trtr: the same 50,000 train images vs 10,000 further TRAIN images (next in the same shuffle), '
                         'two disjoint draws from one distribution, the null of a two-sample test')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--data_root', default=os.path.join(os.environ.get('TMPDIR', '/tmp'), 'ILSVRC', 'Data', 'CLS-LOC'))
    args = ap.parse_args()

    train_dir, val_dir = os.path.join(args.data_root, 'train'), os.path.join(args.data_root, 'val')
    tr_loader, te_loader = make_loaders(train_dir, val_dir, bs=256, workers=args.workers)
    n_tr, n_te = len(tr_loader.dataset), len(te_loader.dataset)
    print(f"device={DEVICE}  bn_mode={args.bn_mode}  train {n_tr} images  val {n_te} images", flush=True)
    if n_tr != N_TRAIN_FULL or n_te != N_VAL_FULL:
        sys.exit(f"REFUSING: expected the full ImageNet ({N_TRAIN_FULL} / {N_VAL_FULL}), got {n_tr} / {n_te}")

    suffix = '' if args.pair == 'trval' else '_trtr'
    out_dir = os.path.join(OUT_ROOT, f'imgnet_rand_{args.bn_mode}{suffix}'); os.makedirs(out_dir, exist_ok=True)
    ref = os.path.join(OUT_ROOT, 'imgnet_simple', 'ep001.npz')          # the fixed splits of every other run
    ref_idx = np.load(ref) if os.path.exists(ref) else None
    print(f"pair {args.pair}: {'50,000 train vs 10,000 validation images' if args.pair == 'trval' else '50,000 train vs 10,000 further train images (disjoint)'}  -> {out_dir}", flush=True)
    for i in range(args.n_init):
        out = os.path.join(out_dir, f'ep{i:03d}.npz')
        if os.path.exists(out):
            print(f"init {i:3d} exists, skip", flush=True); continue
        t0 = time.time()
        torch.manual_seed(args.seed0 + i)
        model = DeepResNet50().to(DEVICE)                                # fresh random weights, no checkpoint
        if args.pair == 'trval':
            h_tr, idx_tr = extract_raw(model, tr_loader, 50000, data_seed=42, bn_mode=args.bn_mode)
            h_te, idx_te = extract_raw(model, te_loader, 10000, data_seed=123, bn_mode=args.bn_mode)
        else:
            # one pass of 60,000 over the seed-42 shuffle: the first 50,000 are exactly the standard training draw
            # (same order, same batches), the next 10,000 are further training images never used as reference
            h_all, idx_all = extract_raw(model, tr_loader, 60000, data_seed=42, bn_mode=args.bn_mode)
            h_tr, idx_tr, h_te, idx_te = h_all[:50000], idx_all[:50000], h_all[50000:], idx_all[50000:]
            assert len(h_te) == 10000 and not (set(idx_te.tolist()) & set(idx_tr.tolist())), "trtr split not disjoint"
        np.savez(out, h_tr=h_tr, h_te=h_te, idx_tr=idx_tr, idx_te=idx_te)
        pair = ''
        if ref_idx is not None:
            same = np.array_equal(ref_idx['idx_tr'], idx_tr) and (args.pair == 'trtr' or np.array_equal(ref_idx['idx_te'], idx_te))
            pair = '  same training images as the checkpoint runs: ' + ('OK' if same else 'MISMATCH')
        fin = np.isfinite(h_tr).all() and np.isfinite(h_te).all()
        print(f"init {i:3d}  seed {args.seed0 + i}  h_tr {h_tr.shape}  |h| median {np.median(np.linalg.norm(h_tr, axis=1)):.3g}  "
              f"finite {fin}  ({time.time() - t0:.0f} s){pair}", flush=True)
        del model
    print("done")


if __name__ == '__main__':
    main()
