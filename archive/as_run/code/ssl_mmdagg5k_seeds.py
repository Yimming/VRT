"""
ssl_mmdagg5k_seeds.py — MMDAgg at 5,000 points per split on the SimCLR models for ADDITIONAL seeds on an explicit
list of epochs (the coarse-to-fine grid already evaluated with seeds 42-51), features extracted on the fly as in
ssl_mmdagg5k_fill.py. One output file per call, sslagg_<dataset>_s<lo>-<hi>_e<epochs>.csv, in the same folder and
format as the 10-seed files (model,epoch,budget,stat,seed,acc,p; p = 0/1 reject flag), so that harvest_ssl.py
reads old and new rows together. Resume: (epoch, seed) pairs present in ANY sslagg file of the dataset are skipped.

  python ssl_mmdagg5k_seeds.py --ckpt_dir <SimCLR ckpt dir> --dataset cifar10 --epochs 450 --seeds 52-91

--bn_train (October 7): features with BatchNorm in train mode (bntrain_extract.extract_bntrain, same protocol as the
CIFAR train-mode features). The label becomes <dataset>_bntrain in the file name and the model column, the resume reads
only rows with that label, and the run must get its own --out_dir (s50/ssl_bntrain), since the eval-mode resume pattern
sslagg_<dataset>_*.csv would also match the train-mode files.
"""
import argparse, os, time, glob, csv
import numpy as np
import torch

from supcon_fill_vrt import SupConResNet, load_ckpt, extract, l2n
from baselines2_stats import mmdagg_test, mmdagg_fn

OUT_DIR = "/home3/p318770/venvs/overfitting/benchmark_final/ssl_baselines"
BUDGET = 5000


def parse_range(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--epochs", required=True, help="explicit epochs, e.g. 450 or 300,310,320 or 461-469")
    ap.add_argument("--seeds", default="52-91")
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--bn_train", action="store_true", help="BatchNorm in train mode during extraction (bntrain_extract)")
    args = ap.parse_args()
    label = f"{args.dataset}_bntrain" if args.bn_train else args.dataset
    if args.bn_train:
        from bntrain_extract import extract_bntrain
        if os.path.abspath(args.out_dir) == os.path.abspath(OUT_DIR):
            raise SystemExit("--bn_train needs its own --out_dir (the eval-mode files live in OUT_DIR)")
    if mmdagg_fn is None:
        raise SystemExit("mmdagg not importable: load the module first, then activate the venv")
    seeds, epochs = parse_range(args.seeds), parse_range(args.epochs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    done = set()                                  # (epoch, seed) already present for this dataset at this budget
    for f in glob.glob(os.path.join(args.out_dir, f"sslagg_{label}_*.csv")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r["budget"] == f"b{BUDGET}" and (not args.bn_train or r["model"] == label):
                    done.add((int(r["epoch"]), int(r["seed"])))
    todo = {ep: [s for s in seeds if (ep, s) not in done] for ep in epochs}
    todo = {ep: ss for ep, ss in todo.items() if ss}
    print(f"{label}: epochs {epochs}, seeds {seeds[0]}-{seeds[-1]}; {sum(len(v) for v in todo.values())} (epoch, seed) pairs to do, device {device}", flush=True)
    if not todo:
        print("nothing to do"); return

    tag = f"s{seeds[0]}-{seeds[-1]}_e{args.epochs.replace(',', '_')}"
    out_csv = os.path.join(args.out_dir, f"sslagg_{label}_{tag}.csv")
    new = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    fout = open(out_csv, "a")
    if new:
        fout.write("model,epoch,budget,stat,seed,acc,p\n"); fout.flush()
    model = SupConResNet(name=args.model).to(device)
    for ep in sorted(todo):
        name = f"ckpt_epoch_{ep}.pth"
        if not os.path.exists(os.path.join(args.ckpt_dir, name)):
            print(f"ep {ep}: no checkpoint {name}, skip", flush=True); continue
        t0 = time.time()
        if args.bn_train:
            feats = extract_bntrain(model, os.path.join(args.ckpt_dir, name), args.dataset, device)
        else:
            load_ckpt(model, os.path.join(args.ckpt_dir, name), device)
            feats = extract(model, args.dataset, device)
        e_tr, e_te = l2n(feats["h_tr"]).astype(np.float32), l2n(feats["h_te"]).astype(np.float32)
        t_feat = time.time() - t0
        rej = []
        for seed in todo[ep]:
            r = mmdagg_test(e_tr, e_te, n_sub=BUDGET, seed=seed); rej.append(r)
            fout.write(f"{label},{ep},b{BUDGET},mmdagg,{seed},nan,{float(r):.1f}\n"); fout.flush()
        print(f"ep {ep:4d}  mmdagg{BUDGET} {sum(rej)}/{len(rej)} reject on the new seeds  (feat {t_feat:.0f} s, total {time.time() - t0:.0f} s)", flush=True)
    fout.close()
    print(f"done -> {out_csv}")


if __name__ == "__main__":
    main()
