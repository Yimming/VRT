"""
moment_diag.py — how far the training and test features of one model differ in their per-dimension mean and variance, at
every saved epoch, under eval-mode extraction and, where the features exist, train-mode BatchNorm extraction. Diagnostic for
the reading that MMD and Energy Distance rely on mean and variance differences between the splits, which train-mode
extraction removes and which models without BatchNorm (ResNet20-FRN, the 5-layer CNN) may not develop.

For each epoch, on the l2-normalized features (the space MMD, ED and VRT work in) and on the unnormalized features:
  mean_ratio   ||mu_a - mu_b||^2 / (tr(S_a)/n_a + tr(S_b)/n_b): about 1 when a and b come from one distribution, larger
               when their means differ (an unbiased signal-to-noise ratio of the mean shift)
  var_lr2      mean over dimensions of log(var_b / var_a)^2: how much the per-dimension variances differ
Each statistic is computed for (a, b) = (all 50,000 training, all 10,000 test features) and, as a null reference, for two
disjoint random subsets of the training features with the same sizes (40,000 and 10,000), which share the split and the
training exposure, so that only sampling noise separates them.

Output: <out_dir>/moment_<tag>.csv with tag, mode, epoch, D, then train-vs-test and null values of both statistics, for the
normalized (_n) and the unnormalized (_r) features. Resume: epochs already in the file are skipped.
Usage:  python moment_diag.py --tag r50 [--feats /scratch/p318770/model_zoos/ladder_feats] --out_dir <BF>/moment_diag
"""
import argparse, csv, glob, os, re, time
import numpy as np


def stats(a, b):
    """mean_ratio and var_lr2 of two feature matrices (rows = points), float64 accumulation"""
    ma, mb = a.mean(0, dtype=np.float64), b.mean(0, dtype=np.float64)
    va, vb = a.var(0, dtype=np.float64, ddof=1), b.var(0, dtype=np.float64, ddof=1)
    mean_ratio = float(((ma - mb) ** 2).sum() / (va.sum() / len(a) + vb.sum() / len(b)))
    eps = 1e-12 * max(float(va.mean()), float(vb.mean()), 1e-30)
    var_lr2 = float(np.mean(np.log((vb + eps) / (va + eps)) ** 2))
    return mean_ratio, var_lr2


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def epochs_in(folder):
    out = []
    for f in glob.glob(os.path.join(folder, "ep*.npz")):
        m = re.fullmatch(r"ep(\d+)\.npz", os.path.basename(f))
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--feats", default="/scratch/p318770/model_zoos/ladder_feats")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"moment_{args.tag}.csv")
    cols = ["tag", "mode", "epoch", "D", "mean_ratio_n", "mean_null_n", "var_lr2_n", "var_null_n",
            "mean_ratio_r", "mean_null_r", "var_lr2_r", "var_null_r"]
    done = set()
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        with open(out_csv) as fh:
            for r in csv.DictReader(fh):
                done.add((r["mode"], int(r["epoch"])))
    new = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    fout = open(out_csv, "a")
    w = csv.writer(fout)
    if new:
        w.writerow(cols); fout.flush()
    for mode, folder in (("eval", os.path.join(args.feats, args.tag)), ("train", os.path.join(args.feats, f"{args.tag}_bntrain"))):
        if not os.path.isdir(folder):
            print(f"{args.tag} {mode}: no features in {folder}, skipped", flush=True); continue
        for ep in epochs_in(folder):
            if (mode, ep) in done:
                continue
            t0 = time.time()
            d = np.load(os.path.join(folder, f"ep{ep:03d}.npz"))
            h_tr, h_te = d["h_tr"].astype(np.float32), d["h_te"].astype(np.float32)
            perm = np.random.default_rng(0).permutation(len(h_tr))
            n_b = min(len(h_te), len(h_tr) // 5)
            ia, ib = perm[n_b:], perm[:n_b]                       # 40,000 and 10,000 when n_tr = 50,000
            row = [args.tag, mode, ep, h_tr.shape[1]]
            for feats in (l2n(h_tr), h_tr):                       # normalized first, then unnormalized
                f_te = l2n(h_te) if feats is not h_tr else h_te
                mr, vl = stats(feats, f_te)
                mr0, vl0 = stats(feats[ia], feats[ib])
                row += [f"{mr:.4f}", f"{mr0:.4f}", f"{vl:.6f}", f"{vl0:.6f}"]
            w.writerow(row); fout.flush()
            print(f"{args.tag} {mode} ep{ep:4d}  mean_ratio_n {row[4]} (null {row[5]})  var_lr2_n {row[6]} (null {row[7]})"
                  f"  ({time.time() - t0:.1f} s)", flush=True)
    fout.close()
    print(f"done -> {out_csv}")


if __name__ == "__main__":
    main()
