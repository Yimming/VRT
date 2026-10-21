"""
cifar_official_check.py — two controls showing whether the high official-split rejection rates of the random-network null
(MMD and ED on CIFAR-10 with ResNet-18, VRT on CIFAR-100) come from the CIFAR images or from the extraction code.
  --features pixels        no network at all: the paper's normalized pixels, 3,072 values per image
  --features r18_shuffled  the SAME random ResNet-18 networks as cifar_null_base.py / cifar_null_vrt.py (seed 1000 + r) and
                           the same analysis seed (100000 + r), but the 60,000 images pass through the network in a random
                           order (new for every repetition) and are put back in the official order afterwards, so position,
                           batch composition and the last partial batch play no role. harvest_official_check.py compares
                           these p-values with the null run repetition by repetition.
Every repetition tests the official split and a random 50,000 / 10,000 split (seed 1000 + r, as in the null runs) with
MMD and ED (10,000 per split, 2,000 permutations, mmd_tst_fast / energy_tst_fast, float32 L2-normalized features) and VRT
(M = 2,000, k = 60, vrt_full on float64 L2-normalized features), the code paths of the null runs. Runs on CPU (torch backend
falls back to the CPU) or GPU.
Output: <out_dir>/official_check_<features>_<dataset>_r<lo>-<hi>.csv   dataset,features,rep,split,method,p
Usage:  python cifar_official_check.py --features pixels --dataset cifar10 --reps 0-19 --out_dir $S50/official_check
"""
import argparse, csv, glob, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cifar_null_vrt import N_TR, N_TE, parse_range, load_cifar, build, set_bn, extract

METHODS = ["mmd", "ed", "vrt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["pixels", "r18_shuffled"], required=True)
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    ap.add_argument("--reps", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    from mmd_tst_fast import mmd_test_fast, l2n as l2n32
    from energy_tst_fast import energy_test_fast
    from vrt_low_lib import vrt_full, l2n as l2n64
    reps = parse_range(args.reps)
    arch = "r18_c10" if args.dataset == "cifar10" else "r18_c100"         # dataset and the paper's normalization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"{args.features}_{args.dataset}" + ("_DRYRUN" if args.dry_run else "")
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"official_check_{tag}_r{reps[0]}-{reps[-1]}.csv")
    done = {}
    for f in glob.glob(os.path.join(args.out_dir, f"official_check_{tag}_r*.csv")):
        if os.path.getsize(f) > 0:
            for r in csv.DictReader(open(f)):
                done.setdefault(int(r["rep"]), set()).add((r["split"], r["method"]))
    need = {(s, m) for s in ("official", "resplit") for m in METHODS}
    todo = [r for r in reps if not need <= done.get(r, set())]
    print(f"{tag}: {len(todo)} of {len(reps)} repetitions to do, device {device}, {torch.get_num_threads()} threads "
          f"-> {out_csv}", flush=True)
    if not todo:
        return
    X = load_cifar(arch, args.dry_run)
    if args.features == "r18_shuffled" and device == "cuda":
        X = X.to(device)
    official = (np.arange(N_TR), np.arange(N_TR, N_TR + N_TE))
    new = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    fout = open(out_csv, "a", newline=""); w = csv.writer(fout)
    if new:
        w.writerow(["dataset", "features", "rep", "split", "method", "p"]); fout.flush()
    for r in todo:
        t0 = time.time()
        seed = 100000 + r
        if args.features == "pixels":
            h = X.reshape(len(X), -1).numpy()                                  # 3,072 normalized pixel values
        else:
            torch.manual_seed(1000 + r)                                        # the network of repetition r
            model, fwd = build(arch)
            model = model.to(device); set_bn(model, "eval")
            order = np.random.default_rng(7_000_000 + r).permutation(N_TR + N_TE)
            h_shuf = extract(model, fwd, X, order, 512, device)
            h = np.empty_like(h_shuf); h[order] = h_shuf                       # back to the official order
            del model
        perm = np.random.default_rng(1000 + r).permutation(N_TR + N_TE)        # the random split of the null runs
        pairs = {"official": official, "resplit": (perm[N_TE:], perm[:N_TE])}
        rows, msg = [], []
        for s in ("official", "resplit"):
            tr, te = pairs[s]
            e_tr, e_te = l2n32(h[tr]), l2n32(h[te])
            n = min(10000, len(e_tr), len(e_te))                               # as energy_tst_fast.main / the null run
            p_mmd = mmd_test_fast(e_tr, e_te, 10000, args.n_perm, seed, "torch")[1]
            p_ed = energy_test_fast(e_tr, e_te, n, args.n_perm, seed, "torch")[1]
            f_tr, f_te = l2n64(h[tr].astype(np.float64)), l2n64(h[te].astype(np.float64))
            p_vrt = vrt_full(f_tr, f_te, f_tr.shape[1], k=60, M=2000, n_qtr=8000, n_perm=args.n_perm, seed=seed)[1]
            for m, p in (("mmd", p_mmd), ("ed", p_ed), ("vrt", p_vrt)):
                rows.append([args.dataset, args.features, r, s, m, f"{p:.6f}"])
            msg.append(f"{s} mmd {p_mmd:.3f} ed {p_ed:.3f} vrt {p_vrt:.3f}")
        w.writerows(rows); fout.flush()
        print(f"rep {r:4d}  " + "  ".join(msg) + f"  ({time.time() - t0:.0f} s)", flush=True)
    fout.close()


if __name__ == "__main__":
    main()
