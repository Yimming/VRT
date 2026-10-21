"""SUPERSEDED (camera-ready code release note). This script belongs to the 10-seed stage of the project and resumes
by counting rows per epoch, which duplicates seeds when an epoch was partly written. The 50-seed results of the
paper come from ssl_s50_fill.py, which resumes by (epoch, seed). Kept for provenance; do not use for new runs."""
"""
ssl_seeds_fill.py — seed-robustness check for the SimCLR onsets at the large budget (10,000 per split):
extra analysis seeds for VRT, MMD and ED on a window of epochs around the onsets, so that the per-epoch
median p-value (and the sustained onset derived from it) can be recomputed with 50 seeds instead of 10.

Same code paths as the published numbers. VRT: supcon_fill_vrt.vrt_test (k=60, M=2,000, 8,000 train queries,
all 10,000 test points, 2,000 permutations), the function that produced the per-epoch SimCLR onsets 189/238.
MMD and ED: mmd_test_fast / energy_test_fast at 10,000 per split, 2,000 permutations, as in ssl_baselines_fill.py.
Features: backbone h, L2-normalized in float32, extracted on the fly per checkpoint.

  python ssl_seeds_fill.py --ckpt_dir <SimCLR ckpt dir> --dataset cifar100 --lo 170 --hi 300 --seeds 52-91

Output: <out_dir>/sslseeds_<dataset>_<lo>-<hi>.csv with model,epoch,budget,stat,seed,acc,p (stat vrt/mmd/ed,
budget b10000, acc nan). Resume: an epoch with 3 x n_seeds rows is skipped. harvest_ssl_seeds.py merges these
with the original 10-seed files.
"""
import argparse, os, time
import numpy as np
import torch

from supcon_fill_vrt import SupConResNet, load_ckpt, extract, l2n, vrt_test
from mmd_tst_fast import mmd_test_fast
from energy_tst_fast import energy_test_fast

OUT_DIR = "/home3/p318770/venvs/overfitting/benchmark_final/ssl_baselines"
BUDGET = 10000
N_PERM = 2000


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
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--seeds", default="52-91", help="extra analysis seeds (the published runs use 42-51)")
    ap.add_argument("--backend", choices=["numpy", "torch"], default="torch")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()
    seeds = parse_range(args.seeds)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = args.backend if device == "cuda" else "numpy"
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"sslseeds_{args.dataset}_{args.lo}-{args.hi}.csv")

    done = {}
    if os.path.exists(out_csv):
        with open(out_csv) as fh:
            next(fh, None)
            for line in fh:
                ep0 = int(line.split(",")[1]); done[ep0] = done.get(ep0, 0) + 1
    done_eps = {e for e, c in done.items() if c >= 3 * len(seeds)}

    names = [f for f in os.listdir(args.ckpt_dir) if f.startswith("ckpt_epoch_") and f.endswith(".pth")]
    names.sort(key=lambda s: int(s[len("ckpt_epoch_"):-len(".pth")]))
    sel = [(int(n[len("ckpt_epoch_"):-len(".pth")]), n) for n in names]
    sel = [(e, n) for e, n in sel if args.lo <= e <= args.hi and e not in done_eps]
    print(f"{args.dataset}: {len(sel)} checkpoints in [{args.lo}, {args.hi}] ({len(done_eps)} already complete), "
          f"seeds {seeds[0]}-{seeds[-1]}, device {device}, backend {backend}", flush=True)

    new = not os.path.exists(out_csv)
    fout = open(out_csv, "a")
    if new:
        fout.write("model,epoch,budget,stat,seed,acc,p\n")
    model = SupConResNet(name=args.model).to(device)
    for ep, name in sel:
        t0 = time.time()
        load_ckpt(model, os.path.join(args.ckpt_dir, name), device)
        feats = extract(model, args.dataset, device)
        e_tr, e_te = l2n(feats["h_tr"]).astype(np.float32), l2n(feats["h_te"]).astype(np.float32)
        D = e_tr.shape[1]
        t_feat = time.time() - t0
        med = {}
        ps = []
        for seed in seeds:
            dc, p = vrt_test(e_tr, e_te, D, k=60, M=2000, n_qtr=8000, n_perm=N_PERM, seed=seed); ps.append(p)
            fout.write(f"{args.dataset},{ep},b{BUDGET},vrt,{seed},nan,{p:.6f}\n")
        med["vrt"] = np.median(ps); ps = []
        for seed in seeds:
            _, p = mmd_test_fast(e_tr, e_te, BUDGET, N_PERM, seed, backend); ps.append(p)
            fout.write(f"{args.dataset},{ep},b{BUDGET},mmd,{seed},nan,{p:.6f}\n")
        med["mmd"] = np.median(ps); ps = []
        for seed in seeds:
            _, p = energy_test_fast(e_tr, e_te, BUDGET, N_PERM, seed, backend); ps.append(p)
            fout.write(f"{args.dataset},{ep},b{BUDGET},ed,{seed},nan,{p:.6f}\n")
        med["ed"] = np.median(ps)
        fout.flush()
        print(f"ep {ep:4d}  median p over {len(seeds)} new seeds: vrt {med['vrt']:.4f}  mmd {med['mmd']:.4f}  "
              f"ed {med['ed']:.4f}  (feat {t_feat:.0f} s, total {time.time() - t0:.0f} s)", flush=True)
    fout.close()
    print(f"done -> {out_csv}")


if __name__ == "__main__":
    main()
