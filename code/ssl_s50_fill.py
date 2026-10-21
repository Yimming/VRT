"""
ssl_s50_fill.py — SimCLR per-epoch statistics for the 50-seed rerun. Features are extracted once per
checkpoint (supcon_fill_vrt: SupConResNet, load_ckpt, extract) and every statistic of the group runs on
them with the same code paths as the published 10-seed numbers:

  --group fast  (GPU, every epoch)   vrt      b10000  supcon_fill_vrt.vrt_test, k=60, M=2,000, 8,000 train
                                                      queries vs all 10,000 test points (paper protocol)
                                     vrt_low  b1500   vrt_low_lib.vrt_low, k=15, M=500, 1,000 vs 1,500
                                                      (rebuttal small-budget protocol, float64 as ksweep15_fill)
                                     mmd, ed  b1500 and b10000 (mmd_test_fast / energy_test_fast, 2,000 perms)
                                     c2st     b1500 and b10000 (baselines2_stats.c2st_test)
  --group slow  (CPU, every 5th)     mmdagg   b1500            (baselines2_stats.mmdagg_test, reject flag)
  --group vrt   (October 8)          vrt and vrt_low only, so VRT can cover every epoch in minutes per epoch
  --group base  (October 8)          mmd, ed, c2st (both budgets; with --bn_train also mmd_raw, ed_raw), the slow part of
                                     "fast", meant for a coarser grid (--every 5) with a refinement near any onset
  --group rev   (October 8)          vrt_rev at 10,000 (M = 2,000, k = 60, 8,000 queries per side) and 1,500 (M = 500,
                                     k = 15, 1,500 per side), the reversed VRT of vrt_reversed_budget.py, same pools, seeds
                                     and permutations as the appendix runs of the other models
  --group knn   (October 8)          schilling and henze at 10,000 and 1,500 per split, k = 3, Henze with 2,000 permutations,
                                     the tests of knn_tst_budget.py (same subsample draws and seeds)
  --group extra                      rev + knn in one pass (one extraction per epoch)

Output: <out_dir>/ssl_<dataset>_<group>_c<lo>-<hi>.csv with model,epoch,budget,stat,seed,acc,p
(acc is nan except for c2st; for mmdagg p holds the 0/1 reject flag); --file_tag adds a suffix to the file name.
Resume: the seeds already present for an (epoch, stat, budget) in ANY ssl_<dataset>_<group>_c*.csv file of out_dir are
skipped, so a run over a new epoch range (for example every epoch, after an every-5th run) computes only what is missing.

--bn_train (October 7): features with BatchNorm in train mode (bntrain_extract.extract_bntrain, same protocol as the
CIFAR train-mode features). The dataset label becomes <dataset>_bntrain in the file names and in the model column, so
train-mode rows never mix with the eval-mode rows; give them their own --out_dir as well (s50/ssl_bntrain).

Unnormalized features (October 7, Table 3): stats mmd_raw and ed_raw are MMD and Energy Distance on the unnormalized
float32 features at 10,000 per split (same tests, same seeds, same permutations as mmd and ed).
  --group raw               only these two stats (eval mode, or train mode with --bn_train)
  --group fast --bn_train   also computes them, so the train-mode SimCLR run writes them in the same pass
Resume reads every ssl_<label>_*_c*.csv file of out_dir (all groups), so raw stats done by one group are not redone by another.
"""
import argparse, os, time, collections, warnings, glob
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)      # one per C2ST fit otherwise

from supcon_fill_vrt import SupConResNet, load_ckpt, extract, l2n, vrt_test
from vrt_low_lib import vrt_low
from mmd_tst_fast import mmd_test_fast
from energy_tst_fast import energy_test_fast
from baselines2_stats import c2st_test, mmdagg_test, mmdagg_fn

N_PERM = 2000
GROUPS = {"fast": [("vrt", 10000), ("vrt_low", 1500), ("mmd", 1500), ("mmd", 10000), ("ed", 1500), ("ed", 10000),
                   ("c2st", 1500), ("c2st", 10000)],
          "vrt": [("vrt", 10000), ("vrt_low", 1500)],                       # VRT alone (minutes per epoch)
          "base": [("mmd", 1500), ("mmd", 10000), ("ed", 1500), ("ed", 10000), ("c2st", 1500), ("c2st", 10000)],
          "slow": [("mmdagg", 1500)],
          "raw": [("mmd_raw", 10000), ("ed_raw", 10000)],
          "rev": [("vrt_rev", 10000), ("vrt_rev", 1500)],                  # test-as-reference VRT (vrt_reversed_budget)
          "knn": [("knn", 10000), ("knn", 1500)],                          # Schilling and Henze, k = 3 (knn_tst_budget)
          "extra": [("vrt_rev", 10000), ("vrt_rev", 1500), ("knn", 10000), ("knn", 1500)]}
# "knn" writes two stats per seed, schilling and henze, from one neighbour graph; its resume check looks at both.
# "vrt" + "base" = "fast"; the split lets VRT run on every epoch while the slow baselines run on a coarser grid (--every)


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
    ap.add_argument("--group", choices=list(GROUPS), required=True)
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--every", type=int, default=1, help="evaluate epochs divisible by this (1 = every epoch)")
    ap.add_argument("--seeds", default="42-91")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--file_tag", default="", help="suffix of the output file name, e.g. _e1 for the every-epoch fill")
    ap.add_argument("--dry_run", action="store_true", help="random features instead of checkpoints (pipeline test)")
    ap.add_argument("--bn_train", action="store_true", help="BatchNorm in train mode during extraction (bntrain_extract)")
    args = ap.parse_args()
    label = f"{args.dataset}_bntrain" if args.bn_train else args.dataset      # file names and model column
    if args.bn_train and not args.dry_run:
        from bntrain_extract import extract_bntrain
    seeds = parse_range(args.seeds)
    stats = list(GROUPS[args.group])
    if args.group in ("fast", "base") and args.bn_train:
        stats += GROUPS["raw"]                                   # train-mode SimCLR: raw MMD and ED in the same pass
    if args.group == "slow" and mmdagg_fn is None:
        raise SystemExit("mmdagg not importable: load the module first, then activate the venv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.group in ("fast", "base", "raw") and device != "cuda" and not args.dry_run:
        raise SystemExit("fast and raw groups need a GPU (published MMD/ED numbers used the torch backend on GPU)")
    backend = "torch" if device == "cuda" else "numpy"
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"ssl_{label}_{args.group}_c{args.lo}-{args.hi}{args.file_tag}.csv")

    if args.dry_run:
        fname = {e: f"ckpt_epoch_{e}.pth" for e in range(args.lo, args.hi + 1)}
    else:
        names = [f for f in os.listdir(args.ckpt_dir) if f.startswith("ckpt_epoch_") and f.endswith(".pth")]
        fname = {int(n[len("ckpt_epoch_"):-len(".pth")]): n for n in names}     # keep the real file names
    eps = sorted(e for e in fname if args.lo <= e <= args.hi and (args.every <= 1 or e % args.every == 0))

    done = collections.defaultdict(lambda: collections.defaultdict(set))     # epoch -> (stat, budget) -> seeds present
    prior = set(glob.glob(os.path.join(args.out_dir, f"ssl_{label}_*_c*.csv")))     # every group of this label
    if os.path.exists(out_csv):
        prior.add(out_csv)
    for fpath in sorted(prior):                                # every file of this dataset and group, own file included
        if os.path.getsize(fpath) == 0:
            continue
        with open(fpath) as fh:
            for line in fh:
                f = line.rstrip("\n").split(",")
                if f[0] != label:
                    continue                                   # header, or rows of another label
                try:
                    done[int(f[1])][(f[3], int(f[2][1:]))].add(int(f[4]))
                except (ValueError, IndexError):
                    continue                                   # header line (possibly repeated)
    def have(e, s, b):                                     # seeds present for (stat, budget); knn needs both of its stats
        if s == "knn":
            return done[e][("henze", b)] & done[e][("schilling", b)]
        return done[e][(s, b)]
    done_eps = {e for e in done if all(set(seeds) <= have(e, s, b) for s, b in stats)}
    todo = [e for e in eps if e not in done_eps]
    print(f"{label} {args.group}: {len(todo)} checkpoints to do in [{args.lo}, {args.hi}] step {args.every} "
          f"({len(done_eps)} complete), seeds {seeds[0]}-{seeds[-1]}, device {device}", flush=True)

    new = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0   # a task killed before its first flush leaves an empty file
    fout = open(out_csv, "a")
    if new:
        fout.write("model,epoch,budget,stat,seed,acc,p\n"); fout.flush()

    def emit(ep, budget, stat, seed, acc, p):
        fout.write(f"{label},{ep},b{budget},{stat},{seed},{acc:.6f},{p:.6f}\n")

    model = None if args.dry_run else SupConResNet(name=args.model).to(device)
    for ep in todo:
        t0 = time.time()
        if args.dry_run:
            rng = np.random.default_rng(ep)
            h_tr = rng.standard_normal((50000, 2048)).astype(np.float32)
            h_te = (rng.standard_normal((10000, 2048)) + 0.002 * ep).astype(np.float32)
        elif args.bn_train:
            feats = extract_bntrain(model, os.path.join(args.ckpt_dir, fname[ep]), args.dataset, device)
            h_tr, h_te = feats["h_tr"], feats["h_te"]
        else:
            load_ckpt(model, os.path.join(args.ckpt_dir, fname[ep]), device)
            feats = extract(model, args.dataset, device)
            h_tr, h_te = feats["h_tr"], feats["h_te"]
        e32_tr, e32_te = l2n(h_tr).astype(np.float32), l2n(h_te).astype(np.float32)   # vrt, mmd, ed, c2st, mmdagg
        r32_tr, r32_te = h_tr.astype(np.float32), h_te.astype(np.float32)             # mmd_raw, ed_raw (unnormalized)
        t_feat = time.time() - t0
        summary = []
        for stat, budget in stats:
            todo_seeds = [s for s in seeds if s not in have(ep, stat, budget)]
            if not todo_seeds:
                continue
            vals = []
            if stat == "vrt":
                D = e32_tr.shape[1]
                for seed in todo_seeds:
                    _, p = vrt_test(e32_tr, e32_te, D, k=60, M=2000, n_qtr=8000, n_perm=N_PERM, seed=seed)
                    emit(ep, budget, stat, seed, np.nan, p); vals.append(p)
            elif stat == "vrt_low":
                e64_tr, e64_te = l2n(h_tr.astype(np.float64)), l2n(h_te.astype(np.float64))
                D = e64_tr.shape[1]
                for seed in todo_seeds:
                    _, p = vrt_low(e64_tr, e64_te, D, seed=seed)
                    emit(ep, budget, stat, seed, np.nan, p); vals.append(p)
            elif stat == "mmd":
                for seed in todo_seeds:
                    _, p = mmd_test_fast(e32_tr, e32_te, budget, N_PERM, seed, backend)
                    emit(ep, budget, stat, seed, np.nan, p); vals.append(p)
            elif stat == "ed":
                for seed in todo_seeds:
                    _, p = energy_test_fast(e32_tr, e32_te, budget, N_PERM, seed, backend)
                    emit(ep, budget, stat, seed, np.nan, p); vals.append(p)
            elif stat == "mmd_raw":
                for seed in todo_seeds:
                    _, p = mmd_test_fast(r32_tr, r32_te, budget, N_PERM, seed, backend)
                    emit(ep, budget, stat, seed, np.nan, p); vals.append(p)
            elif stat == "ed_raw":
                for seed in todo_seeds:
                    _, p = energy_test_fast(r32_tr, r32_te, budget, N_PERM, seed, backend)
                    emit(ep, budget, stat, seed, np.nan, p); vals.append(p)
            elif stat == "vrt_rev":
                from vrt_reversed_budget import vrt_reversed, BUDGETS as REV_BUDGETS
                tr_pool, te_pool, M, k, n_q = REV_BUDGETS["large" if budget == 10000 else "small"]
                X_tr, X_te = l2n(h_tr.astype(np.float64)), l2n(h_te.astype(np.float64))     # float64 as the appendix runs
                D = X_tr.shape[1]
                for seed in todo_seeds:
                    rng = np.random.RandomState(1000 + seed)                                # budget subsample draw
                    tr = X_tr[rng.permutation(len(X_tr))[:tr_pool]]
                    te = X_te[rng.permutation(len(X_te))[:te_pool]]
                    _, p = vrt_reversed(tr, te, D, M, k, n_q, N_PERM, seed)
                    emit(ep, budget, stat, seed, np.nan, p); vals.append(p)
            elif stat == "knn":
                from knn_tst_budget import knn_graph, schilling_p, henze_p
                K = 3
                X_tr, X_te = l2n(h_tr.astype(np.float32)), l2n(h_te.astype(np.float32))     # float32 as knn_tst_budget
                p_sch = []
                for seed in todo_seeds:
                    rng = np.random.default_rng(seed)
                    n = min(budget, len(X_tr), len(X_te))
                    X = X_tr[rng.choice(len(X_tr), n, replace=False)]
                    Y = X_te[rng.choice(len(X_te), n, replace=False)]
                    Z = np.vstack([X, Y]); tags = np.array([0] * n + [1] * n)
                    idx = knn_graph(Z, K, "torch" if device == "cuda" else "sklearn")
                    p_s = schilling_p(idx, tags, n, n, K)
                    p_h = henze_p(idx, tags, N_PERM, seed)
                    emit(ep, budget, "schilling", seed, np.nan, p_s); emit(ep, budget, "henze", seed, np.nan, p_h)
                    vals.append(p_h); p_sch.append(p_s)
            elif stat == "c2st":
                accs = []
                for seed in todo_seeds:
                    acc, p = c2st_test(e32_tr, e32_te, n_sub=budget, seed=seed)
                    emit(ep, budget, stat, seed, acc, p); vals.append(p); accs.append(acc)
            elif stat == "mmdagg":
                for seed in todo_seeds:
                    r = mmdagg_test(e32_tr, e32_te, n_sub=budget, seed=seed)
                    emit(ep, budget, stat, seed, np.nan, float(r)); vals.append(float(r))
            fout.flush()
            if stat == "mmdagg":
                summary.append(f"mmdagg{budget} {int(sum(vals))}/{len(vals)}")
            elif stat == "c2st":
                summary.append(f"c2st{budget} p{np.median(vals):.3f} acc{np.median(accs):.3f}")
            elif stat == "knn":
                summary.append(f"henze{budget} {np.median(vals):.4f} schilling{budget} {np.median(p_sch):.4f}")
            else:
                summary.append(f"{stat}{budget} {np.median(vals):.4f}")
        print(f"ep {ep:4d}  " + "  ".join(summary) + f"   (feat {t_feat:.0f} s, total {time.time() - t0:.0f} s)", flush=True)
    fout.close()
    print(f"done -> {out_csv}")


if __name__ == "__main__":
    main()
