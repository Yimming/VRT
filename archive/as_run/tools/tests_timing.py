"""
tests_timing.py — wall-clock time per test and per seed, one machine, one feature set (camera-ready #27102, October 15).

Same idea as vrt_timing.py (synthetic data, VRT only) extended to every test of the benchmark, with the code paths of
the 50-seed runs (ssl_s50_fill.py / base_fill.py / the standalone scripts) so that the numbers describe what was run:
  vrt       b10000  vrt_low_lib.vrt_full         k = 60, M = 2,000, 8,000 train queries, all 10,000 test queries, 2,000 perms,
                                                 float64 features as vrt_forward_budget.py (the CIFAR models of Table 1)
  vrt       b1500   vrt_low_lib.vrt_low          k = 15, M = 500, 1,000 vs 1,500, float64, 2,000 perms
  vrt32     b10000  supcon_fill_vrt.vrt_test     the same test on float32 features, the SimCLR path of ssl_s50_fill.py
  mmd       b10000, b1500   mmd_tst_fast.mmd_test_fast      Gaussian kernel, median heuristic, 2,000 permutations
  ed        b10000, b1500   energy_tst_fast.energy_test_fast  2,000 permutations
  c2st      b10000, b1500   baselines2_stats.c2st_test      sklearn MLP (20 hidden units, 100 iterations), CPU
  mmdagg    b5000, b1500    baselines2_stats.mmdagg_test    mmdagg package, wild bootstrap, CPU
  knn       b10000, b1500   knn_tst_budget: one k = 3 neighbour graph, Schilling (normal approximation) and Henze
                            (2,000 permutations); the time covers both tests together, as in the runs
  vrt_rev   b10000, b1500   vrt_reversed_budget.vrt_reversed (with --rev only)
Feature extraction is not timed (the runs extract once per checkpoint for all tests). Features: an npz of one checkpoint
(--npz, keys h_tr/h_te or X_tr/X_te as the stored CIFAR features) or synthetic points on the sphere (--synthetic D,
train seed 0 and test seed 1, as vrt_timing.py). l2-normalized float32 for the kernel, C2ST, MMDAgg and kNN tests and
for vrt32; float64 for vrt b10000, vrt b1500 and vrt_rev (both budgets), as vrt_forward_budget.py / vrt_reversed_budget.py.
Protocol (October 16, second version): a fixed number of CPU threads for torch and BLAS (--threads, default the SLURM
allocation; set through OMP/OPENBLAS/MKL variables and threadpoolctl, and printed with the BLAS library found), one
untimed warm-up run per test (--warmup seed, default 41), then the timed seeds interleaved seed after seed so that the
methods alternate in time (--no_interleave times test after test). The CPU-only comparison uses --backend numpy, which
hides any GPU (CUDA_VISIBLE_DEVICES is emptied before torch loads): MMD and ED in NumPy, the kNN graph with the blocked
Gram-matrix search of knn_graph's torch backend on the CPU, which is the implementation the experiments used on the GPU;
--knn_sklearn times scikit-learn's NearestNeighbors instead (a different implementation, about 300 s per seed at 10,000
points in the first run). --knn_check compares, for the first seed, the neighbour sets of the two graphs and, with
--knn_ref, the Schilling and Henze p-values of every timed seed with the stored GPU results of knn_v2 (same draw and
the same algorithm; float32 near-ties in the Gram-matrix search can differ between CPU and GPU, so a few neighbours and
the p-values may differ slightly, which the check reports). --mmdagg_jax adds the JAX implementation of the mmdagg package on
the CPU (float64 unless --jax_f32, same subsample as mmdagg_test; the warm-up run compiles it).
Output: a CSV with one row per (test, budget, seed) and
a table of median / min / max seconds per seed with the implied time per epoch for 50 seeds. Both models of the paper's
table run in one job so that they share the machine (timing_cpu_tasks.txt).
Usage (release root, venv active; the script adds code/ to the path):
  python tools/tests_timing.py --npz /scratch/p318770/model_zoos/ladder_feats/r50/ep061.npz --tag r50_ep61 --seeds 42-51 \
      --backend numpy --rev --out $S50/timing/tests_timing_r50_ep61_cpu8.csv
  --only mmd,ed      run only these tests;   --skip mmdagg5000,c2st10000   skip some;   --rev   add the reversed VRT
"""
import argparse, csv, os, platform, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))   # release tree: the runners live in code/

N_PERM = 2000
KEY_PAIRS = [("h_tr", "h_te"), ("X_tr", "X_te"), ("tr", "te"), ("train", "test"), ("emb_tr", "emb_te")]


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def load_npz(path):
    d = np.load(path)
    for a, b in KEY_PAIRS:
        if a in d.files and b in d.files:
            return d[a], d[b]
    raise KeyError(f"unknown keys in {path}: {d.files}")


def sample_sphere(n, D, seed):
    rng = np.random.RandomState(seed)
    x = rng.standard_normal((n, D))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


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
    ap.add_argument("--npz", default="")
    ap.add_argument("--synthetic", type=int, default=0, help="dimension of synthetic sphere data (50,000 train, 10,000 test)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default="42-44")
    ap.add_argument("--backend", choices=["torch", "numpy"], default="torch", help="mmd, ed and the kNN graph")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--rev", action="store_true")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for torch and BLAS (0 = SLURM_CPUS_PER_TASK or all)")
    ap.add_argument("--warmup", type=int, default=41, help="seed of one untimed run per test before the timed seeds (-1 = none)")
    ap.add_argument("--no_interleave", action="store_true", help="time test after test instead of seed after seed")
    ap.add_argument("--knn_sklearn", action="store_true", help="build the kNN graph with scikit-learn instead of the blocked Gram-matrix search (knn_graph torch backend, CPU or GPU)")
    ap.add_argument("--knn_check", action="store_true", help="after timing: compare the Gram-matrix graph with scikit-learn's for the first seed, and the Schilling and Henze p-values of every seed with --knn_ref")
    ap.add_argument("--knn_ref", default="", help="glob of knn_v2 CSVs of the same model (tag,n_samples,k,epoch,seed,p_schilling,p_henze) for --knn_check")
    ap.add_argument("--epoch", type=int, default=-1, help="epoch of the npz (default: parsed from its name, ep061 -> 61); used by --knn_check")
    ap.add_argument("--mmdagg_jax", action="store_true", help="also time the JAX implementation of the mmdagg package on the CPU (x64 unless --jax_f32)")
    ap.add_argument("--jax_f32", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = parse_range(a.seeds)
    if a.backend == "numpy":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU-only run: no library may pick up a GPU (knn_graph would)
    if a.mmdagg_jax:
        os.environ.setdefault("XLA_FLAGS", f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={a.threads or os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count())}")

    n_thr = a.threads or int(os.environ.get("SLURM_CPUS_PER_TASK", 0) or os.cpu_count())
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n_thr)                     # takes effect for libraries loaded below
    import torch
    torch.set_num_threads(n_thr)
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
        threadpool_limits(n_thr)
        blas = ", ".join(f"{i['internal_api']} {i.get('version', '?')} x{i['num_threads']}" for i in threadpool_info())
    except ImportError:
        blas = "threadpoolctl not installed, BLAS threads set through the environment variables only"
    from supcon_fill_vrt import vrt_test
    from vrt_low_lib import vrt_low, vrt_full
    from mmd_tst_fast import mmd_test_fast
    from energy_tst_fast import energy_test_fast
    from baselines2_stats import c2st_test, mmdagg_test
    from knn_tst_budget import knn_graph, schilling_p, henze_p
    cuda = torch.cuda.is_available() and a.backend == "torch"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CUDA_VISIBLE_DEVICES='')" if a.backend == "numpy" else "none"
    print(f"[{a.tag}] host {platform.node()}, cpus {os.cpu_count()} (SLURM {os.environ.get('SLURM_CPUS_PER_TASK', '-')}), "
          f"threads {n_thr} (torch {torch.get_num_threads()}; {blas}), gpu {gpu}, backend {a.backend}, seeds {seeds}, "
          f"warm-up seed {a.warmup if a.warmup >= 0 else 'none'}, {'test after test' if a.no_interleave else 'seed after seed (interleaved)'}", flush=True)

    if a.npz:
        h_tr, h_te = load_npz(a.npz)
    elif a.synthetic:
        h_tr, h_te = sample_sphere(50000, a.synthetic, 0), sample_sphere(10000, a.synthetic, 1)
    else:
        sys.exit("give --npz or --synthetic D")
    e32_tr, e32_te = l2n(h_tr.astype(np.float32)), l2n(h_te.astype(np.float32))
    e64_tr, e64_te = l2n(h_tr.astype(np.float64)), l2n(h_te.astype(np.float64))
    D = e32_tr.shape[1]
    print(f"  features {e32_tr.shape[0]} train, {e32_te.shape[0]} test, D = {D}", flush=True)

    knn_p = {}                                           # (budget, seed) -> (p_schilling, p_henze) of the timed graph

    def knn_pool(budget, seed):
        rng = np.random.default_rng(seed)                # the draw of knn_tst_budget.py and ssl_s50_fill.py
        n = min(budget, len(e32_tr), len(e32_te))
        X = e32_tr[rng.choice(len(e32_tr), n, replace=False)]
        Y = e32_te[rng.choice(len(e32_te), n, replace=False)]
        return np.vstack([X, Y]), np.array([0] * n + [1] * n), n

    def knn(budget, seed):
        Z, tags, n = knn_pool(budget, seed)
        idx = knn_graph(Z, 3, "sklearn" if a.knn_sklearn else "torch")   # torch backend: CPU here (no GPU visible)
        res = (schilling_p(idx, tags, n, n, 3), henze_p(idx, tags, N_PERM, seed))
        knn_p[(budget, seed)] = res
        return res

    import re
    epoch = a.epoch if a.epoch >= 0 else int(m.group(1)) if (m := re.search(r"ep(\d+)", os.path.basename(a.npz))) else -1

    mmdagg_jax_fn = None
    if a.mmdagg_jax:
        import jax
        if not a.jax_f32:
            jax.config.update("jax_enable_x64", True)
        from mmdagg.jax import mmdagg as mmdagg_jax_fn
        print(f"  mmdagg JAX {jax.__version__}, devices {jax.devices()}, x64 {not a.jax_f32}", flush=True)

    def mmdagg_jax_test(budget, seed):
        """same subsample as baselines2_stats.mmdagg_test, run through the JAX implementation"""
        rng = np.random.RandomState(seed)
        X = e32_tr[rng.permutation(len(e32_tr))[:budget]]
        Y = e32_te[rng.permutation(len(e32_te))[:budget]]
        if not a.jax_f32:
            X, Y = X.astype(np.float64), Y.astype(np.float64)
        try:
            r = mmdagg_jax_fn(X, Y, seed=seed)
        except TypeError:
            r = mmdagg_jax_fn(X, Y, key=jax.random.PRNGKey(seed))
        return float(int(r))                              # int() waits for the computation to finish

    tests = [
        ("vrt", 10000, lambda s: vrt_full(e64_tr, e64_te, D, k=60, M=2000, n_qtr=8000, n_perm=N_PERM, seed=s)[1]),   # CIFAR path: float64, vrt_forward_budget.py
        ("vrt", 1500, lambda s: vrt_low(e64_tr, e64_te, D, seed=s)[1]),                                               # float64, vrt_forward_budget.py small budget
        ("vrt32", 10000, lambda s: vrt_test(e32_tr, e32_te, D, k=60, M=2000, n_qtr=8000, n_perm=N_PERM, seed=s)[1]),  # SimCLR path: float32, ssl_s50_fill.py
        ("mmd", 10000, lambda s: mmd_test_fast(e32_tr, e32_te, 10000, N_PERM, s, a.backend)[1]),
        ("mmd", 1500, lambda s: mmd_test_fast(e32_tr, e32_te, 1500, N_PERM, s, a.backend)[1]),
        ("ed", 10000, lambda s: energy_test_fast(e32_tr, e32_te, 10000, N_PERM, s, a.backend)[1]),
        ("ed", 1500, lambda s: energy_test_fast(e32_tr, e32_te, 1500, N_PERM, s, a.backend)[1]),
        ("c2st", 10000, lambda s: c2st_test(e32_tr, e32_te, n_sub=10000, seed=s)[1]),
        ("c2st", 1500, lambda s: c2st_test(e32_tr, e32_te, n_sub=1500, seed=s)[1]),
        ("knn", 10000, lambda s: knn(10000, s)[1]),
        ("knn", 1500, lambda s: knn(1500, s)[1]),
        ("mmdagg", 1500, lambda s: float(mmdagg_test(e32_tr, e32_te, n_sub=1500, seed=s))),
        ("mmdagg", 5000, lambda s: float(mmdagg_test(e32_tr, e32_te, n_sub=5000, seed=s))),
    ]
    if mmdagg_jax_fn is not None:
        tests += [("mmdagg_jax", 1500, lambda s: mmdagg_jax_test(1500, s)), ("mmdagg_jax", 5000, lambda s: mmdagg_jax_test(5000, s))]
    if a.rev:
        from vrt_reversed_budget import vrt_reversed, BUDGETS as REV_BUDGETS

        def rev(budget, s):
            tr_pool, te_pool, M, k, n_q = REV_BUDGETS["large" if budget == 10000 else "small"]
            rng = np.random.RandomState(1000 + s)
            tr = e64_tr[rng.permutation(len(e64_tr))[:tr_pool]]
            te = e64_te[rng.permutation(len(e64_te))[:te_pool]]
            return vrt_reversed(tr, te, D, M, k, n_q, N_PERM, s)[1]
        tests += [("vrt_rev", 10000, lambda s: rev(10000, s)), ("vrt_rev", 1500, lambda s: rev(1500, s))]
    only = set(a.only.split(",")) - {""}
    skip = set(a.skip.split(",")) - {""}
    tests = [t for t in tests if (not only or t[0] in only or f"{t[0]}{t[1]}" in only)
             and t[0] not in skip and f"{t[0]}{t[1]}" not in skip]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    device_of = {(n, b): (("cuda" if cuda else "cpu") if n in ("mmd", "ed", "knn") else "cpu") for n, b, _ in tests}
    print(f"  kNN graph: {'scikit-learn brute force' if a.knn_sklearn else 'blocked Gram-matrix search and top-k (knn_graph torch backend)'} on "
          f"{'the GPU' if cuda else 'the CPU'}", flush=True)

    def timed(fn, s):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        r = fn(s)
        if cuda:
            torch.cuda.synchronize()
        return time.time() - t0, r

    if a.warmup >= 0:                                    # one untimed run per test (library loading, caches, JIT)
        t0 = time.time()
        for name, budget, fn in tests:
            fn(a.warmup)
        print(f"  warm-up with seed {a.warmup} done ({time.time() - t0:.0f} s)", flush=True)

    ts = {(n, b): [] for n, b, _ in tests}
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "test", "budget", "backend", "device", "threads", "seed", "seconds", "result"])
        order = ([(s, t) for s in seeds for t in tests] if not a.no_interleave         # seed-major: the methods alternate
                 else [(s, t) for t in tests for s in seeds])
        for s, (name, budget, fn) in order:
            dt, r = timed(fn, s)
            ts[(name, budget)].append(dt)
            w.writerow([a.tag, name, budget, a.backend, device_of[(name, budget)], n_thr, s, f"{dt:.3f}", f"{r:.6f}"])
            f.flush()
            if not a.no_interleave and (name, budget) == (tests[-1][0], tests[-1][1]):
                print(f"  seed {s} done: " + "  ".join(f"{n}{b} {ts[(n, b)][-1]:.1f}s" for n, b, _ in tests), flush=True)
    print(f"\n{'test':8s} {'budget':>7s} {'device':6s} {'s per seed, median':>19s} {'min':>9s} {'max':>9s} {'per epoch (50 seeds)':>21s}")
    for name, budget, _ in tests:
        v = ts[(name, budget)]
        m = 50 * np.median(v)
        per = f"{m / 3600:.2f} h" if m >= 3600 else f"{m / 60:.1f} min" if m >= 60 else f"{m:.0f} s"
        print(f"{name:8s} {budget:7d} {device_of[(name, budget)]:6s} {np.median(v):19.2f} {min(v):9.2f} {max(v):9.2f} {per:>21s}")
    print(f"written {a.out}")

    if a.knn_check and any(n == "knn" for n, _, _ in tests):
        print(f"\nkNN check (epoch {epoch}):")
        for budget in (10000, 1500):
            if (budget, seeds[0]) not in knn_p:
                continue
            Z, tags, n = knn_pool(budget, seeds[0])
            idx_t = knn_graph(Z, 3, "torch"); idx_s = knn_graph(Z, 3, "sklearn")
            same = np.mean([set(r1) == set(r2) for r1, r2 in zip(idx_t, idx_s)])
            ps_s, ph_s = schilling_p(idx_s, tags, n, n, 3), henze_p(idx_s, tags, N_PERM, seeds[0])
            ps_t, ph_t = knn_p[(budget, seeds[0])]
            print(f"  b{budget} seed {seeds[0]}: Gram-matrix graph against scikit-learn, identical neighbour sets in {100 * same:.3f}% of the "
                  f"{len(Z)} rows; p Schilling {ps_t:.6f} vs {ps_s:.6f}, Henze {ph_t:.4f} vs {ph_s:.4f}")
        if a.knn_ref:
            import glob
            ref = {}
            for f in glob.glob(a.knn_ref):
                for r in csv.DictReader(open(f)):
                    try:
                        ref[(int(r["n_samples"]), int(r["epoch"]), int(r["seed"]))] = (float(r["p_schilling"]), float(r["p_henze"]))
                    except (KeyError, ValueError):
                        continue
            for budget in (10000, 1500):
                pairs = [(knn_p[(budget, s_)], ref[(budget, epoch, s_)]) for s_ in seeds if (budget, s_) in knn_p and (budget, epoch, s_) in ref]
                if not pairs:
                    print(f"  b{budget}: no reference rows for epoch {epoch} in {a.knn_ref}"); continue
                ds = max(abs(x[0] - y[0]) for x, y in pairs); dh = max(abs(x[1] - y[1]) for x, y in pairs)
                print(f"  b{budget}: {len(pairs)} seeds against the stored GPU results (knn_v2), max |dp| Schilling {ds:.6f}, Henze {dh:.4f}"
                      + ("  (identical within the printed precision)" if ds < 1e-6 and dh < 1e-4 else "  (DIFFERENT)"))


if __name__ == "__main__":
    main()
