"""
mmd_on_chat_s50.py — MMD on VRT's per-point c-hat scores under the unified protocol (Appendix I, camera-ready, October 12).

The submitted Appendix I (Figure 9) ran MMD on the c-hat values of one seed (42), with the methods_v2 scoring, a
3,000-point subsample per side and a biased statistic (mmd_on_chat.py, discussion_data/mmd_on_chat_*.log). This rerun uses
the s50 protocol:
  c-hat     for every analysis seed 42 to 91, exactly the values of the s50 large-budget VRT: the draw of
            vrt_low_lib.vrt_full (one permutation of the training split; M = 2,000 reference points, the next 8,000
            training points and all 10,000 test points as queries; k = 60; float64 l2-normalized features)
  MMD       mmd_tst_fast.mmd_test_fast on all 8,000 + 10,000 scalar c-hat values (Gaussian kernel, median heuristic,
            unbiased statistic, 2,000 permutations, seed = analysis seed), the MMD of Table 1 applied to c-hat
  check     the VRT median difference dc of the same draw is written too; it must equal the dc column of
            $S50/vrt_fwd/vrt_fwd_<tag>_large_c*.csv for the same epoch and seed
CSV: tag,epoch,seed,dc,mmd_c,p    (resume on (epoch, seed); one file per call, named by --out)
Usage (GPU node, venv active, s50 code folder, which holds vrt_low_lib.py and mmd_tst_fast.py):
  python mmd_on_chat_s50.py --tag r50 --npz_fmt '/scratch/p318770/model_zoos/ladder_feats/r50/ep{ep:03d}.npz' \
      --epochs 0-9 --out /home3/p318770/venvs/overfitting/benchmark_final/s50/mmd_on_chat/mmdchat_r50_c0-9.csv
"""
import argparse, csv, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vrt_low_lib import compute_chat, l2n
from mmd_tst_fast import mmd_test_fast

M, K, N_QTR = 2000, 60, 8000
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


def parse_range(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def chat_vrt_full(e_tr, e_te, D, seed):
    """the c-hat values of vrt_low_lib.vrt_full for this seed (same RandomState draw), and its dc"""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(e_tr))
    ref = e_tr[perm[:M]]
    q_tr = e_tr[perm[M:M + min(N_QTR, len(e_tr) - M)]]
    c_tr = compute_chat(q_tr, ref, D, K)
    c_te = compute_chat(e_te, ref, D, K)
    return c_tr, c_te, float(np.median(c_te) - np.median(c_tr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--npz_fmt", required=True)
    ap.add_argument("--epochs", required=True)
    ap.add_argument("--seeds", default="42-91")
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--backend", choices=["torch", "numpy"], default="torch")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seeds, epochs = parse_range(args.seeds), parse_range(args.epochs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = set()
    if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        with open(args.out) as f:
            done = {(int(r["epoch"]), int(r["seed"])) for r in csv.DictReader(f)}
    new = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    fout = open(args.out, "a", newline=""); w = csv.writer(fout)
    if new:
        w.writerow(["tag", "epoch", "seed", "dc", "mmd_c", "p"]); fout.flush()
    print(f"[{args.tag}] MMD on c-hat: M={M}, k={K}, {N_QTR} train vs all test queries, MMD on all c-hat values, "
          f"{args.n_perm} permutations, seeds {seeds[0]}-{seeds[-1]}, backend {args.backend}", flush=True)
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
        ps = []
        for s in todo:
            c_tr, c_te, dc = chat_vrt_full(e_tr, e_te, D, s)
            # n_sub = 10,000 caps a side at the protocol budget; with 8,000 + 10,000 values the draws and the result are
            # identical to using all values (the r50 and A runs of October 12), larger test sets are subsampled to 10,000
            stat, p = mmd_test_fast(c_tr[:, None], c_te[:, None], n_sub=10000,
                                    n_perm=args.n_perm, seed=s, backend=args.backend)
            w.writerow([args.tag, ep, s, f"{dc:+.6f}", f"{stat:.8e}", f"{p:.6f}"]); ps.append(p)
        fout.flush()
        print(f"  ep{ep:4d}  median p {np.median(ps):.4f}  {sum(p < 0.05 for p in ps)}/{len(ps)} seeds < 0.05  "
              f"({time.time() - t0:.0f} s)", flush=True)
    fout.close(); print("done", flush=True)


if __name__ == "__main__":
    main()
