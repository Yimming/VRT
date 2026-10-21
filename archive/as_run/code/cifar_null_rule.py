"""
cifar_null_rule.py — blank control of the paper's decision rule on CIFAR: the paper's own VRT procedure, unchanged, applied
to networks that have learned nothing (an "epoch 0" with random weights).

For network i: torch.manual_seed(net_seed0 + i), a fresh model of the chosen architecture with default initialization
(no checkpoint), features of the 60,000 CIFAR images extracted as in cifar_null_vrt.py (eval mode, the paper's
preprocessing). Then exactly what the paper does at every checkpoint:
  the OFFICIAL split (official 50,000 training images vs official 10,000 test images)
  VRT with the paper's 50 analysis seeds 42 to 91, i.e. the same draws of training images as at every checkpoint of the
  paper's CIFAR models (the 10,000 test images are the same for every seed)
    large  M = 2,000, k = 60, 8,000 training queries vs all 10,000 test images   (vrt_low_lib.vrt_full; the SimCLR runs
           used the same function on float32 features, kept here for the simclr architectures)
    small  M = 500, k = 15, 1,000 vs 1,500 queries                               (vrt_low_lib.vrt_low)
  decision: median p over the 50 seeds below 0.05
The harvest (harvest_cifar_null_rule.py) counts, per architecture, the networks the rule would flag as a detection.
Output: <out_dir>/vrt_rule_<arch>_eval_n<lo>-<hi>.csv, one row per (network, budget, seed):
  arch,bn_mode,net,net_seed,budget,seed,dc,p
Resume: a network with all its (budget, seed) rows in any vrt_rule_<arch>_eval_*.csv of out_dir is skipped; the rows of one
network are written together.
Usage:  python cifar_null_rule.py --arch r18_c100 --nets 0-9 --out_dir $S50/null_cifar_rule
"""
import argparse, csv, glob, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cifar_null_vrt import ARCHS, N_TR, N_TE, parse_range, load_cifar, build, set_bn, extract
from vrt_low_lib import vrt_full, vrt_low, l2n

PAPER_SEEDS = list(range(42, 92))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCHS), required=True)
    ap.add_argument("--nets", required=True, help="network indices, e.g. 0-9")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--net_seed0", type=int, default=5000, help="network i: torch.manual_seed(net_seed0 + i)")
    ap.add_argument("--seeds", default="42-91", help="analysis seeds (the paper's: 42-91)")
    ap.add_argument("--budgets", default="large,small")
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    nets, seeds, budgets = parse_range(args.nets), parse_range(args.seeds), args.budgets.split(",")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not args.dry_run:
        raise SystemExit("needs a GPU for the feature extraction")
    bs = ARCHS[args.arch][2]
    tag = f"{args.arch}_eval" + ("_DRYRUN" if args.dry_run else "")
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"vrt_rule_{tag}_n{nets[0]}-{nets[-1]}.csv")
    done = {}
    for f in glob.glob(os.path.join(args.out_dir, f"vrt_rule_{tag}_n*.csv")):
        if os.path.getsize(f) == 0:
            continue
        for r in csv.DictReader(open(f)):
            done.setdefault(int(r["net"]), set()).add((r["budget"], int(r["seed"])))
    need = {(b, s) for b in budgets for s in seeds}
    todo = [i for i in nets if not need <= done.get(i, set())]
    print(f"{tag}: {len(todo)} of {len(nets)} networks to do ({nets[0]}-{nets[-1]}), official split, seeds "
          f"{seeds[0]}-{seeds[-1]}, budgets {budgets}, device {torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'} "
          f"-> {out_csv}", flush=True)
    if not todo:
        return
    X = load_cifar(args.arch, args.dry_run)
    if device == "cuda":
        X = X.to(device)
    new = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    fout = open(out_csv, "a", newline=""); w = csv.writer(fout)
    if new:
        w.writerow(["arch", "bn_mode", "net", "net_seed", "budget", "seed", "dc", "p"]); fout.flush()
    flagged = {b: [0, 0] for b in budgets}
    for i in todo:
        t0 = time.time()
        torch.manual_seed(args.net_seed0 + i)
        model, fwd = build(args.arch)
        model = model.to(device); set_bn(model, "eval")
        h = extract(model, fwd, X, np.arange(N_TR + N_TE), bs, device)
        h_tr, h_te = h[:N_TR], h[N_TR:]                                     # the official split
        t_feat = time.time() - t0
        if not (np.isfinite(h_tr).all() and np.isfinite(h_te).all()):
            print(f"net {i}: NON-FINITE FEATURES, skipped", flush=True); continue
        e64_tr, e64_te = l2n(h_tr.astype(np.float64)), l2n(h_te.astype(np.float64))
        if args.arch.startswith("simclr"):                                   # ssl_s50_fill: float32 for the large VRT
            e_lg_tr, e_lg_te = l2n(h_tr).astype(np.float32), l2n(h_te).astype(np.float32)
        else:                                                                # vrt_forward_budget: float64
            e_lg_tr, e_lg_te = e64_tr, e64_te
        D = h.shape[1]
        rows, msg = [], []
        for b in budgets:
            ps = []
            for s in seeds:
                if b == "large":
                    dc, p = vrt_full(e_lg_tr, e_lg_te, D, k=60, M=2000, n_qtr=8000, n_perm=args.n_perm, seed=s)
                else:
                    dc, p = vrt_low(e64_tr, e64_te, D, k=15, M=500, n_qtr=1000, n_qte=1500, n_perm=args.n_perm, seed=s)
                rows.append([args.arch, "eval", i, args.net_seed0 + i, b, s, f"{dc:+.6e}", f"{p:.6f}"]); ps.append(p)
            med = float(np.median(ps)); hit = med < 0.05
            flagged[b][0] += hit; flagged[b][1] += 1
            msg.append(f"{b}: median p {med:.3f}, {sum(p < 0.05 for p in ps)}/{len(ps)} seeds < 0.05"
                       + ("  FLAGGED" if hit else ""))
        w.writerows(rows); fout.flush()
        print(f"net {i:3d}  " + "   ".join(msg) + f"  (features {t_feat:.0f} s, total {time.time() - t0:.0f} s)", flush=True)
        del model
    fout.close()
    print("this task, networks flagged by the paper's rule:  " + "  ".join(f"{b} {k}/{n}" for b, (k, n) in flagged.items()),
          flush=True)


if __name__ == "__main__":
    main()
