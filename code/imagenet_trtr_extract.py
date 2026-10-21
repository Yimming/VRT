"""
imagenet_trtr_extract.py — train-vs-train features of the TRAINED ImageNet checkpoints: the standard 50,000 training
images (same draw, same batches as every other ImageNet run) against 10,000 FURTHER training images, extracted in
one pass of 60,000 over the seed-42 shuffle, so that the second sample has the same batch structure as the
validation sample of the standard files (its own batches of 256 from one split). Both splits come from the training
distribution and both were seen in training, so a two-sample test has nothing to detect: rejections on these files
measure what the extraction procedure itself (eval-mode or train-mode BatchNorm) and the test produce under the null,
checkpoint by checkpoint. Output
    ladder_feats/imgnet_<scheme>[_bntrain]_trtr/ep{ep:03d}.npz   (h_tr 50,000, h_te 10,000, idx_tr, idx_te)
The 50,000 are compared with the stored standard file of the checkpoint (indices must match; the feature difference
is printed and should be at GPU rounding level). Requires the full ImageNet under $TMPDIR/ILSVRC/Data/CLS-LOC.
  python imagenet_trtr_extract.py --scheme both --bn_mode eval
  python imagenet_trtr_extract.py --scheme both --bn_mode train
"""
import os, sys, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imagenet_extract import (DeepResNet50, make_loaders, extract_raw, load_checkpoint, set_bn_mode,
                              CKPT_DIR, EPOCHS, OUT_ROOT, DEVICE, N_TRAIN_FULL, N_VAL_FULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scheme', required=True, choices=['simple', 'advanced', 'both'])
    ap.add_argument('--bn_mode', choices=['eval', 'train'], default='eval')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--data_root', default=os.path.join(os.environ.get('TMPDIR', '/tmp'), 'ILSVRC', 'Data', 'CLS-LOC'))
    ap.add_argument('--epochs', default=None, help="comma-separated checkpoint subset, default all nine")
    args = ap.parse_args()
    epochs = [int(e) for e in args.epochs.split(',')] if args.epochs else EPOCHS

    tr_loader, te_loader = make_loaders(os.path.join(args.data_root, 'train'), os.path.join(args.data_root, 'val'),
                                        bs=256, workers=args.workers)
    n_tr, n_te = len(tr_loader.dataset), len(te_loader.dataset)
    print(f"device={DEVICE}  bn_mode={args.bn_mode}  train {n_tr} images  val {n_te} images", flush=True)
    if n_tr != N_TRAIN_FULL or n_te != N_VAL_FULL:
        sys.exit(f"REFUSING: expected the full ImageNet ({N_TRAIN_FULL} / {N_VAL_FULL}), got {n_tr} / {n_te}")

    schemes = ['simple', 'advanced'] if args.scheme == 'both' else [args.scheme]
    model = DeepResNet50().to(DEVICE)
    print(f"BatchNorm layers switched to batch statistics: {set_bn_mode(model, args.bn_mode)}", flush=True)
    for scheme in schemes:
        base_tag = ('imgnet_simple' if scheme == 'simple' else 'imgnet_adv') + ('_bntrain' if args.bn_mode == 'train' else '')
        out_dir = os.path.join(OUT_ROOT, base_tag + '_trtr'); os.makedirs(out_dir, exist_ok=True)
        for ep in epochs:
            out = os.path.join(out_dir, f'ep{ep:03d}.npz')
            if os.path.exists(out):
                print(f"[{base_tag}_trtr] ep{ep:3d} exists, skip", flush=True); continue
            ckpt = os.path.join(CKPT_DIR, f'resnet50_{scheme}_pretraining_checkpoint_epoch_{ep}.pt')
            if not os.path.exists(ckpt):
                print(f"[{base_tag}_trtr] ep{ep:3d} checkpoint not found: {ckpt}", flush=True); continue
            t0 = time.time()
            load_checkpoint(model, ckpt)
            h_all, idx_all = extract_raw(model, tr_loader, 60000, data_seed=42, bn_mode=args.bn_mode)
            h_tr, idx_tr, h_te, idx_te = h_all[:50000], idx_all[:50000], h_all[50000:], idx_all[50000:]
            assert len(h_te) == 10000 and not (set(idx_te.tolist()) & set(idx_tr.tolist())), "split not disjoint"
            np.savez(out, h_tr=h_tr, h_te=h_te, idx_tr=idx_tr, idx_te=idx_te)
            note = ''
            ref = os.path.join(OUT_ROOT, base_tag, f'ep{ep:03d}.npz')
            if os.path.exists(ref):
                r = np.load(ref)
                same = np.array_equal(r['idx_tr'], idx_tr)
                diff = float(np.abs(r['h_tr'] - h_tr).max()) if same else float('nan')
                note = f"  standard 50,000: indices {'OK' if same else 'MISMATCH'}, max |feature diff| {diff:.3g}"
            print(f"[{base_tag}_trtr] ep{ep:3d}  h_tr {h_tr.shape} h_te {h_te.shape}  ({time.time() - t0:.0f} s){note}", flush=True)
    print("done")


if __name__ == '__main__':
    main()
