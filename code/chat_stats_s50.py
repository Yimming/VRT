"""
chat_stats_s50.py — which part of the c-hat distribution moves first (camera-ready #27102, October 14).

Appendix I shows that MMD on the c-hat scores rejects at the same epoch as VRT or earlier. VRT compares the medians of
the two c-hat samples; MMD compares their whole distributions. This script recomputes, for every analysis seed, exactly
the c-hat values of the s50 large-budget VRT (vrt_low_lib.vrt_full: RandomState(seed), M = 2,000 reference points and
the next 8,000 training points as queries from one permutation of the training split, all 10,000 test points as
queries, k = 60, float64 l2-normalized features) and then runs ONE permutation loop with the SAME permutation stream as
vrt_full (the 2,000 draws of rng.permutation after the reference draw), evaluating several statistics on every
permutation. The MMD-on-c-hat run (mmd_on_chat_s50.py) used the permutation stream of mmd_test_fast, which differs from
this one; only the nine statistics below share permutations with VRT. Statistics:
  med    median(c_te) - median(c_tr)                 VRT itself; its p must equal vrt_fwd_<tag>_large_c*.csv (checked)
  mean   mean difference
  tmean  10 percent trimmed mean difference
  q10, q25, q75, q90   differences of the quantiles
  sd     difference of the standard deviations (spread)
  ks     Kolmogorov-Smirnov distance (one-sided, larger is more extreme)
All two-sided except ks: p = (#{|stat_perm| >= |stat_obs|} + 1) / (n_perm + 1).
It also writes the per-seed summaries of c_tr and c_te (median, mean, sd, quantiles) so that the direction of each change
can be read off, and stores the c-hat arrays of the check seed (42) per epoch in <out>_chat_s42.npz (float64) for histograms.
CSV: tag,epoch,seed,stat,obs,p  plus rows stat = tr_<summary> / te_<summary> with obs = value and p empty.
Usage (CPU task, venv active, s50 code folder):
  python chat_stats_s50.py --tag v7 --npz_fmt '/scratch/p318770/model_zoos/ladder_feats/v7/ep{ep:03d}.npz' \
      --epochs 15-45 --out $S50/chat_stats/chatstats_v7_c15-45.csv
"""
import argparse, csv, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vrt_low_lib import compute_chat, l2n

M, K, N_QTR = 2000, 60, 8000
KEY_PAIRS = [("h_tr", "h_te"), ("X_tr", "X_te"), ("tr", "te"), ("train", "test"), ("emb_tr", "emb_te")]
QS = (0.10, 0.25, 0.75, 0.90)
STATS = ["med", "mean", "tmean", "q10", "q25", "q75", "q90", "sd", "ks"]


def load_epoch(fmt, ep):
    f = fmt.format(ep=ep)
    if not os.path.exists(f):
        return None
    d = np.load(f)
    for a, b in KEY_PAIRS:
        if a in d.files and b in d.files:
            return d[a], d[b]
    raise KeyError(f"unknown keys in {f}: {d.files}")


def parse_range(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def summaries(c):
    q = np.quantile(c, QS)
    return {"median": float(np.median(c)), "mean": float(c.mean()), "sd": float(c.std()),
            "q10": float(q[0]), "q25": float(q[1]), "q75": float(q[2]), "q90": float(q[3])}


def stat_vector(a, b):
    """all statistics of sample b (test) against sample a (train), in the order of STATS"""
    sa, sb = np.sort(a), np.sort(b)
    na, nb = len(sa), len(sb)
    ta, tb = int(0.1 * na), int(0.1 * nb)
    qa, qb = np.quantile(sa, QS), np.quantile(sb, QS)
    # KS distance from the sorted samples
    allv = np.concatenate([sa, sb])
    fa = np.searchsorted(sa, allv, side="right") / na
    fb = np.searchsorted(sb, allv, side="right") / nb
    ks = float(np.max(np.abs(fb - fa)))
    return np.array([np.median(sb) - np.median(sa), sb.mean() - sa.mean(),
                     sb[tb:nb - tb].mean() - sa[ta:na - ta].mean(),
                     qb[0] - qa[0], qb[1] - qa[1], qb[2] - qa[2], qb[3] - qa[3],
                     sb.std() - sa.std(), ks])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--npz_fmt", required=True)
    ap.add_argument("--epochs", required=True)
    ap.add_argument("--seeds", default="42-91")
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--check_seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds, epochs = parse_range(args.seeds), parse_range(args.epochs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = set()
    if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        with open(args.out) as f:
            done = {(int(r["epoch"]), int(r["seed"])) for r in csv.DictReader(f) if r["stat"] == "ks"}
    new = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    fout = open(args.out, "a", newline=""); w = csv.writer(fout)
    if new:
        w.writerow(["tag", "epoch", "seed", "stat", "obs", "p"]); fout.flush()
    npz_path = os.path.splitext(args.out)[0] + "_chat_s%d.npz" % args.check_seed
    saved = dict(np.load(npz_path)) if os.path.exists(npz_path) else {}
    print(f"[{args.tag}] c-hat statistics: M={M}, k={K}, {N_QTR} train vs all test queries, {args.n_perm} permutations "
          f"(same stream as vrt_full), seeds {seeds[0]}-{seeds[-1]}, stats {STATS}", flush=True)
    for ep in epochs:
        todo = [s for s in seeds if (ep, s) not in done]
        if not todo:
            continue
        loaded = load_epoch(args.npz_fmt, ep)
        if loaded is None:
            continue
        t0 = time.time()
        e_tr, e_te = l2n(loaded[0].astype(np.float64)), l2n(loaded[1].astype(np.float64))
        D = e_tr.shape[1]
        pmed = []
        for s in todo:
            rng = np.random.RandomState(s)                              # vrt_full: reference draw, then the permutations
            perm = rng.permutation(len(e_tr))
            ref = e_tr[perm[:M]]
            q_tr = e_tr[perm[M:M + min(N_QTR, len(e_tr) - M)]]
            c_tr = compute_chat(q_tr, ref, D, K)
            c_te = compute_chat(e_te, ref, D, K)
            obs = stat_vector(c_tr, c_te)
            pooled = np.concatenate([c_tr, c_te]); na = len(c_tr)
            count = np.zeros(len(STATS))
            for _ in range(args.n_perm):
                p = rng.permutation(len(pooled))                        # identical draws to vrt_full
                sv = stat_vector(pooled[p[:na]], pooled[p[na:]])
                count[:-1] += np.abs(sv[:-1]) >= np.abs(obs[:-1])
                count[-1] += sv[-1] >= obs[-1]
            pv = (count + 1) / (args.n_perm + 1)
            for name, o, pp in zip(STATS, obs, pv):
                w.writerow([args.tag, ep, s, name, f"{o:+.6f}", f"{pp:.6f}"])
            for side, c in (("tr", c_tr), ("te", c_te)):
                for name, val in summaries(c).items():
                    w.writerow([args.tag, ep, s, f"{side}_{name}", f"{val:.6f}", ""])
            pmed.append(pv[0])
            if s == args.check_seed:
                saved[f"ep{ep:03d}_tr"] = c_tr.astype(np.float64); saved[f"ep{ep:03d}_te"] = c_te.astype(np.float64)
        fout.flush()
        np.savez(npz_path, **saved)
        print(f"  ep{ep:4d}  median p of the median statistic {np.median(pmed):.4f}  ({time.time() - t0:.0f} s)", flush=True)
    fout.close(); print("done", flush=True)


if __name__ == "__main__":
    main()
