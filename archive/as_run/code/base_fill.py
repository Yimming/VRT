"""
base_fill.py — C2ST and MMDAgg on precomputed per-epoch features (camera-ready unified protocol; the published runs
used --seeds 42-91, 50 seeds, the default 42-51 is the 10-seed default of the earlier stage).
Functions are imported verbatim from baselines2_stats.py (the rebuttal script, itself verbatim from the
supplementary methods.py): c2st_test (sklearn MLP, 20 ReLU units, normal-approximation p, held-out acc)
and mmdagg_test (mmdagg.np, B1 = B2 = 2,000, binary reject).

  python base_fill.py --stat c2st   --tag r50 --npz_fmt ".../r50/ep{ep:03d}.npz" --budgets 1500,10000 --every 1
  python base_fill.py --stat mmdagg --tag r50 --npz_fmt ".../r50/ep{ep:03d}.npz" --budgets 1500 --every 1
  python base_fill.py --stat mmdagg --tag r50 --npz_fmt ".../r50/ep{ep:03d}.npz" --budgets 5000 --every 5

Output CSV <out_dir>/<stat>_<tag>_b<budgets>.csv with model,epoch,budget,stat,seed,acc,p (acc is nan for
mmdagg, whose p column is the 0/1 reject flag). Resume: an (epoch, budget) pair with all seeds present is skipped.
"""
import argparse, os, time, csv
import numpy as np
from baselines2_stats import c2st_test, mmdagg_test, mmdagg_fn
from release_utils import validate_features, ResumeIndex, open_append

KEY_PAIRS = [("h_tr", "h_te"), ("X_tr", "X_te"), ("tr", "te"), ("train", "test"), ("emb_tr", "emb_te")]


def load_epoch(fmt, ep):
    f = fmt.format(ep=ep)
    if not os.path.exists(f):
        return None
    d = np.load(f)
    for a, b in KEY_PAIRS:
        if a in d.files and b in d.files:
            return d[a], d[b]
    raise KeyError(f"unknown keys in {f}: {d.files}")


def l2n(x):
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


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
    ap.add_argument("--stat", choices=["c2st", "mmdagg"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--npz_fmt", required=True)
    ap.add_argument("--budgets", default="1500,10000")
    ap.add_argument("--every", type=int, default=1, help="evaluate epochs divisible by N (1 = every saved epoch)")
    ap.add_argument("--seeds", default="42-51")
    ap.add_argument("--epochs", default="0-1000")
    ap.add_argument("--raw", action="store_true", help="skip L2 normalization (raw feature space, Appendix A-4)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_suffix", default="", help="appended to the CSV name, so that array tasks on disjoint epochs of one tag write separate files")
    args = ap.parse_args()

    if args.stat == "mmdagg" and mmdagg_fn is None:
        raise SystemExit("mmdagg package not importable in this environment")
    budgets = [int(b) for b in args.budgets.split(",")]
    seeds = parse_range(args.seeds); epochs = parse_range(args.epochs)
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"{args.stat}_{args.tag}_b{'-'.join(map(str, budgets))}{args.out_suffix}.csv")

    # release fix: resume by (epoch, budget, seed) instead of a row count per (epoch, budget); the as-run version
    # reran every seed of an incomplete (epoch, budget) and duplicated the seeds already present
    # Valid rows only: p in [0, 1] (for mmdagg the 0/1 decision), acc in [0, 1] for c2st and 'nan' for mmdagg; rows
    # of another model or statistic are ignored.
    # The MMDAgg decision must be 0 or 1 (discrete), C2ST p is a normal approximation in [0, 1]; a row of another model
    # or statistic makes the resume refuse. The "features" column records the normalization (as-run files lack it).
    features = "raw" if args.raw else "l2"
    HEADER = ["model", "epoch", "budget", "stat", "seed", "acc", "p", "features"]
    index = ResumeIndex(out_csv, key_cols=("epoch", "budget", "seed"), value_cols=("acc", "p"), int_cols=("epoch", "seed"),
                        ranges={"p": (0, 1), "acc": (0, 1)}, nan_ok=("acc",) if args.stat == "mmdagg" else (),
                        discrete={"p": {0.0, 1.0}} if args.stat == "mmdagg" else None,
                        expect={"model": args.tag, "stat": args.stat, "features": features}, header=HEADER)
    print(index.report())
    fout, _w = open_append(out_csv, HEADER)
    print(f"[{args.tag}] {args.stat}, budgets {budgets}, every {args.every}, seeds {seeds[0]}-{seeds[-1]}, "
          f"{'RAW (un-normalized)' if args.raw else 'L2-normalized'} features", flush=True)

    for ep in epochs:
        if args.every > 1 and ep % args.every != 0:
            continue
        todo_seeds = {b: [s for s in seeds if not index.has(ep, f"b{b}", s)] for b in budgets}
        todo = [b for b in budgets if todo_seeds[b]]
        if not todo:
            continue
        loaded = load_epoch(args.npz_fmt, ep)
        if loaded is None:
            continue
        validate_features(*loaded, names=("train features", "test features"), allow_zero_rows=args.raw)
        if min(len(loaded[0]), len(loaded[1])) < max(todo):   # release fix: never silently test a smaller budget
            raise SystemExit(f"ep {ep}: split sizes {len(loaded[0])}/{len(loaded[1])} are below the budget {max(todo)}")
        t0 = time.time()
        if args.raw:                                          # a zero row is a valid point of the raw feature space
            e_tr, e_te = loaded[0].astype(np.float32), loaded[1].astype(np.float32)
        else:
            e_tr, e_te = l2n(loaded[0]), l2n(loaded[1])
        summary = []
        for b in todo:
            vals = []
            for seed in todo_seeds[b]:
                if args.stat == "c2st":
                    acc, p = c2st_test(e_tr, e_te, n_sub=b, seed=seed)
                    fout.write(f"{args.tag},{ep},b{b},c2st,{seed},{acc:.6f},{p:.6f},{features}\n"); vals.append((acc, p))
                else:
                    rej = mmdagg_test(e_tr, e_te, n_sub=b, seed=seed)
                    fout.write(f"{args.tag},{ep},b{b},mmdagg,{seed},nan,{float(rej):.1f},{features}\n"); vals.append((np.nan, float(rej)))
            fout.flush()
            if args.stat == "c2st":
                summary.append(f"b{b}: med_p={np.median([p for _, p in vals]):.3f} med_acc={np.median([a for a, _ in vals]):.3f}")
            else:
                summary.append(f"b{b}: {int(sum(p for _, p in vals))}/{len(vals)} reject")
        print(f"  ep{ep:4d}  " + "  ".join(summary) + f"  ({time.time() - t0:.0f} s)", flush=True)
    fout.close(); print("done")


if __name__ == "__main__":
    main()
