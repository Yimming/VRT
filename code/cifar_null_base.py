"""
cifar_null_base.py — Type-I check of the baselines on CIFAR images through randomly initialized networks, the companion of
cifar_null_vrt.py. Repetition r uses the SAME network (seed0 + r), the SAME random 50,000 / 10,000 split of the 60,000 CIFAR
images (seed0 + r) and the SAME analysis seed (vrt_seed0 + r) as repetition r of the VRT run, and every method is called
through the unchanged s50 code path, at the 10,000 tier of Table 1:
  --part gpu   MMD (mmd_tst_fast, 10,000 per split, 2,000 permutations, torch backend)
               ED (energy_tst_fast, 10,000 per split, 2,000 permutations, torch backend)
               Henze and Schilling (knn_tst_budget, 10,000 per split, k = 3, 2,000 permutations, torch backend)
               features of all 60,000 images are extracted on the GPU (eval mode)
  --part cpu   C2ST (baselines2_stats.c2st_test, 10,000 per split) and MMDAgg (baselines2_stats.mmdagg_test, 5,000 per
               split), as base_fill.py. Only the images these two tests draw are passed through the network, on the CPU:
               the drawn rows are placed at their positions in a full-size training array, so c2st_test and mmdagg_test
               make exactly the draws they make on stored features.
Features L2-normalized in float32 as in the s50 scripts. With the same analysis seed, VRT (large), MMD, ED and C2ST use the
same 10,000 training images and all 10,000 test images, MMDAgg the first 5,000 of them; Henze/Schilling draw their own.
--splits resplit,official : the random split (independent repetitions, the Type-I estimate) and the official split
(official test images in every repetition, for comparison). Default: both for --part gpu, resplit only for --part cpu;
--part cpu --splits official adds C2ST and MMDAgg on the official split (file name ends in _official).
Output: <out_dir>/base_null_<arch>_<bn_mode>_<part>_r<lo>-<hi>[_<splits>].csv
  arch,bn_mode,rep,net_seed,split_seed,seed,split,method,budget,p,acc,sec
  (MMDAgg: p is the 0/1 reject flag at alpha 0.05, as in base_fill.py; acc only for C2ST)
Resume: a repetition with all its (split, method) rows in any base_null_<arch>_<bn_mode>_<part>_*.csv is skipped; the rows
of one repetition are written together.
Usage:  python cifar_null_base.py --part gpu --arch r18_c10 --reps 0-99 --out_dir $S50/null_cifar_base
        python cifar_null_base.py --part cpu --methods c2st,mmdagg --arch r18_c10 --reps 0-9 --out_dir $S50/null_cifar_base
"""
import argparse, csv, glob, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cifar_null_vrt import ARCHS, N_TR, N_TE, parse_range, load_cifar, build, set_bn, extract

N_TIER = 10000          # Table 1 budget per split; MMDAgg at 5,000
N_AGG = 5000
PARTS = {"gpu": ["mmd", "ed", "henze", "schilling"], "cpu": ["c2st", "mmdagg"]}
BUDGET = {"mmd": N_TIER, "ed": N_TIER, "henze": N_TIER, "schilling": N_TIER, "c2st": N_TIER, "mmdagg": N_AGG}


def run_gpu_methods(e_tr, e_te, seed, n_perm, methods):
    from mmd_tst_fast import mmd_test_fast
    from energy_tst_fast import energy_test_fast
    from knn_tst_budget import knn_graph, schilling_p, henze_p
    out = {}
    n = min(N_TIER, len(e_tr), len(e_te))
    if "mmd" in methods:
        out["mmd"] = (mmd_test_fast(e_tr, e_te, N_TIER, n_perm, seed, "torch")[1], np.nan)
    if "ed" in methods:
        out["ed"] = (energy_test_fast(e_tr, e_te, n, n_perm, seed, "torch")[1], np.nan)
    if "henze" in methods or "schilling" in methods:            # knn_tst_budget.main, one seed
        rng = np.random.default_rng(seed)
        X = e_tr[rng.choice(len(e_tr), n, replace=False)]
        Y = e_te[rng.choice(len(e_te), n, replace=False)]
        Z = np.vstack([X, Y]); tags = np.array([0] * n + [1] * n)
        idx = knn_graph(Z, 3, "torch")
        out["schilling"] = (schilling_p(idx, tags, n, n, 3), np.nan)
        out["henze"] = (henze_p(idx, tags, n_perm, seed), np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=list(PARTS), required=True)
    ap.add_argument("--methods", default=None, help="subset of the part's methods, comma separated (default: all)")
    ap.add_argument("--arch", choices=list(ARCHS), required=True)
    ap.add_argument("--bn_mode", choices=["eval"], default="eval")
    ap.add_argument("--reps", required=True)
    ap.add_argument("--splits", default=None, help="resplit,official (default: both for gpu, resplit for cpu)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--vrt_seed0", type=int, default=100000)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    methods = args.methods.split(",") if args.methods else PARTS[args.part]
    assert set(methods) <= set(PARTS[args.part]), f"{methods} not in part {args.part}"
    splits = (args.splits or ("resplit,official" if args.part == "gpu" else "resplit")).split(",")
    reps = parse_range(args.reps)
    if args.part == "gpu":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda" and not args.dry_run:
            raise SystemExit("--part gpu needs a GPU (the s50 MMD/ED/Henze numbers used the torch backend on the GPU)")
    else:
        device = "cpu"
        if "mmdagg" in methods:
            from baselines2_stats import mmdagg_fn
            if mmdagg_fn is None:
                raise SystemExit("mmdagg not importable: load the module first, then activate the venv")
    tag = f"{args.arch}_{args.bn_mode}_{args.part}" + ("_DRYRUN" if args.dry_run else "")
    os.makedirs(args.out_dir, exist_ok=True)
    # a run with explicitly chosen splits writes its own file (e.g. ..._r0-9_official.csv), so it never appends to the file
    # of a run that may still be going; resume and harvest read every base_null_<tag>_r*.csv
    extra = "" if args.splits is None else "_" + "-".join(splits)
    out_csv = os.path.join(args.out_dir, f"base_null_{tag}_r{reps[0]}-{reps[-1]}{extra}.csv")

    done = {}
    for f in glob.glob(os.path.join(args.out_dir, f"base_null_{tag}_r*.csv")):
        if os.path.getsize(f) == 0:
            continue
        for r in csv.DictReader(open(f)):
            done.setdefault(int(r["rep"]), set()).add((r["split"], r["method"]))
    need = {(s, m) for s in splits for m in methods}
    todo = [r for r in reps if not need <= done.get(r, set())]
    print(f"{tag}: methods {methods}, splits {splits}: {len(todo)} of {len(reps)} repetitions to do "
          f"({reps[0]}-{reps[-1]}), device {torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}, "
          f"{torch.get_num_threads()} threads -> {out_csv}", flush=True)
    if not todo:
        return

    X = load_cifar(args.arch, args.dry_run)
    if device == "cuda":
        X = X.to(device)
    bs = ARCHS[args.arch][2]
    official = (np.arange(N_TR), np.arange(N_TR, N_TR + N_TE))
    new = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    fout = open(out_csv, "a", newline=""); w = csv.writer(fout)
    if new:
        w.writerow(["arch", "bn_mode", "rep", "net_seed", "split_seed", "seed", "split", "method", "budget", "p", "acc", "sec"])
        fout.flush()
    rej = {(s, m): [0, 0] for s in splits for m in methods}
    for r in todo:
        t0 = time.time()
        net_seed, split_seed, seed = args.seed0 + r, args.seed0 + r, args.vrt_seed0 + r
        torch.manual_seed(net_seed)
        model, fwd = build(args.arch)
        model = model.to(device); set_bn(model, "eval")
        perm = np.random.default_rng(split_seed).permutation(N_TR + N_TE)        # as cifar_null_vrt.py
        resplit = (perm[N_TE:], perm[:N_TE])
        pairs = {"resplit": resplit, "official": official}
        rows, msg = [], []
        if args.part == "gpu":
            from mmd_tst_fast import l2n
            h = extract(model, fwd, X, np.arange(N_TR + N_TE), bs, device)
            t_feat = time.time() - t0
            for s in splits:
                tr, te = pairs[s]
                e_tr, e_te = l2n(h[tr]), l2n(h[te])
                t1 = time.time()
                res = run_gpu_methods(e_tr, e_te, seed, args.n_perm, methods)
                for m in methods:
                    p, acc = res[m]
                    rows.append([args.arch, args.bn_mode, r, net_seed, split_seed, seed, s, m, BUDGET[m],
                                 f"{p:.6f}", "nan", f"{time.time() - t1:.0f}"])
                    rej[(s, m)][0] += p < 0.05; rej[(s, m)][1] += 1
                msg.append(s + " " + " ".join(f"{m} {res[m][0]:.3f}" for m in methods))
        else:
            from baselines2_stats import c2st_test, mmdagg_test
            from base_fill import l2n
            t_feat = 0.0
            for s in splits:
                tr, te = pairs[s]
                # the training rows c2st_test draws (RandomState(seed).permutation(len(e_tr))[:10,000]); mmdagg_test
                # draws the first 5,000 of the same permutation; both use all or part of the 10,000 test images
                pos = np.random.RandomState(seed).permutation(len(tr))[:N_TIER]
                t1 = time.time()
                hf = extract(model, fwd, X, np.concatenate([tr[pos], te]), bs, device)
                t_feat += time.time() - t1
                D = hf.shape[1]
                # rows the tests should never draw are NaN, so a draw outside `pos` would fail loudly instead of
                # silently testing empty features
                e_tr = np.full((len(tr), D), np.nan, np.float32); e_tr[pos] = hf[:len(pos)]
                with np.errstate(invalid="ignore"):
                    e_tr, e_te = l2n(e_tr), l2n(hf[len(pos):])
                assert np.all(np.linalg.norm(e_tr[pos], axis=1) > 0.5)
                for m in methods:
                    t1 = time.time()
                    if m == "c2st":
                        acc, p = c2st_test(e_tr, e_te, n_sub=N_TIER, seed=seed)
                    else:
                        acc, p = np.nan, float(mmdagg_test(e_tr, e_te, n_sub=N_AGG, seed=seed))
                    rows.append([args.arch, args.bn_mode, r, net_seed, split_seed, seed, s, m, BUDGET[m],
                                 f"{p:.6f}", f"{acc:.6f}" if m == "c2st" else "nan", f"{time.time() - t1:.0f}"])
                    hit = (p >= 0.5) if m == "mmdagg" else (p < 0.05)
                    rej[(s, m)][0] += hit; rej[(s, m)][1] += 1
                    msg.append(f"{s} {m} " + (f"p {p:.3f} acc {acc:.3f}" if m == "c2st" else f"reject {int(p)}"))
        w.writerows(rows); fout.flush()
        print(f"rep {r:4d}  {'  '.join(msg)}  (features {t_feat:.0f} s, total {time.time() - t0:.0f} s)", flush=True)
        del model
    fout.close()
    print("this task, single-test rejections at 0.05:  " +
          "  ".join(f"{s} {m} {k}/{n}" for (s, m), (k, n) in rej.items()), flush=True)


if __name__ == "__main__":
    main()
