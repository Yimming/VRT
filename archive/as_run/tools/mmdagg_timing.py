"""
mmdagg_timing.py — MMDAgg, NumPy against JAX implementation of the authors' package, on one CPU node (camera-ready
#27102, October 17). Standalone: needs only numpy and the mmdagg package (with jax for the JAX side), so it runs in a
separate environment (installing jax into the experiment venv would upgrade its numpy and scipy, which we do not touch).

Same subsample as baselines2_stats.mmdagg_test: RandomState(seed), the first n_sub rows of a permutation of each split,
float32, l2-normalized features of one checkpoint (--npz). Settings: the package defaults (Laplace and Gaussian kernels,
10 bandwidths, uniform weights, B1 = B2 = 2,000, B3 = 50, alpha 0.05), the same for both implementations; JAX in float64
(x64 enabled) unless --jax_f32, the same precision as the NumPy path (its distance matrices are float64). One untimed
warm-up run per implementation and budget (seed 41; the JAX run compiles there), then the timed seeds interleaved
(seed-major). Threads: --threads for XLA's intra-op pool and the BLAS variables, set before the libraries load.
Output: CSV tag,impl,budget,seed,seconds,reject and a table of median / min / max seconds per seed.
Usage (cluster, code folder, the jaxtime venv active):
  python mmdagg_timing.py --npz /scratch/p318770/model_zoos/ladder_feats/r50/ep061.npz --tag r50_ep61 --seeds 42-51 \
      --threads 8 --out $S50/timing/mmdagg_np_jax_r50_ep61.csv
"""
import argparse, csv, os, platform, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default="42-51")
    ap.add_argument("--budgets", default="1500,5000")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=41)
    ap.add_argument("--jax_f32", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(a.threads)
    os.environ.setdefault("XLA_FLAGS", f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={a.threads}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import numpy as np
    import jax
    if not a.jax_f32:
        jax.config.update("jax_enable_x64", True)
    import mmdagg
    from mmdagg.np import mmdagg as agg_np
    from mmdagg.jax import mmdagg as agg_jax
    try:
        from importlib.metadata import version
        ver = version("mmdagg")
    except Exception:
        ver = "?"
    seeds = []
    for part in a.seeds.split(","):
        lo, hi = (part.split("-") + [part])[:2]; seeds += list(range(int(lo), int(hi) + 1))
    budgets = [int(b) for b in a.budgets.split(",")]
    print(f"[{a.tag}] host {platform.node()}, threads {a.threads}, numpy {np.__version__}, jax {jax.__version__} on {jax.devices()}, "
          f"mmdagg {ver}, x64 {not a.jax_f32}, seeds {seeds}, budgets {budgets}, warm-up seed {a.warmup}", flush=True)

    d = np.load(a.npz)
    keys = [(x, y) for x, y in (("h_tr", "h_te"), ("X_tr", "X_te"), ("tr", "te"), ("train", "test")) if x in d.files and y in d.files][0]
    h_tr, h_te = d[keys[0]].astype(np.float32), d[keys[1]].astype(np.float32)
    e_tr = h_tr / np.maximum(np.linalg.norm(h_tr, axis=1, keepdims=True), 1e-12)
    e_te = h_te / np.maximum(np.linalg.norm(h_te, axis=1, keepdims=True), 1e-12)
    print(f"  features {e_tr.shape[0]} train, {e_te.shape[0]} test, D = {e_tr.shape[1]}", flush=True)

    def draw(n_sub, seed):                                   # baselines2_stats.mmdagg_test
        rng = np.random.RandomState(seed)
        X = e_tr[rng.permutation(len(e_tr))[:n_sub]].astype(np.float32)
        Y = e_te[rng.permutation(len(e_te))[:n_sub]].astype(np.float32)
        return X, Y

    def run_np(n_sub, seed):
        X, Y = draw(n_sub, seed)
        return int(agg_np(X, Y, seed=seed))

    def run_jax(n_sub, seed):
        X, Y = draw(n_sub, seed)
        if not a.jax_f32:
            X, Y = X.astype(np.float64), Y.astype(np.float64)
        try:
            r = agg_jax(X, Y, seed=seed)
        except TypeError:
            r = agg_jax(X, Y, key=jax.random.PRNGKey(seed))
        return int(r)                                        # blocks until the computation is done

    impls = [("numpy", run_np), ("jax", run_jax)]
    cells = [(name, b, fn) for b in budgets for name, fn in impls]
    if a.warmup >= 0:
        t0 = time.time()
        for name, b, fn in cells:
            t1 = time.time(); fn(b, a.warmup); print(f"  warm-up {name} b{b}: {time.time() - t1:.1f} s (JAX includes compilation)", flush=True)
        print(f"  warm-up done ({time.time() - t0:.0f} s)", flush=True)
    ts = {(n, b): [] for n, b, _ in cells}; dec = {(n, b): [] for n, b, _ in cells}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tag", "impl", "budget", "seed", "seconds", "reject"])
        for s in seeds:
            for name, b, fn in cells:
                t0 = time.time(); r = fn(b, s); dt = time.time() - t0
                ts[(name, b)].append(dt); dec[(name, b)].append(r); w.writerow([a.tag, name, b, s, f"{dt:.3f}", r]); f.flush()
            print(f"  seed {s}: " + "  ".join(f"{n} b{b} {ts[(n, b)][-1]:.1f}s" for n, b, _ in cells), flush=True)
    print(f"\n{'impl':6s} {'budget':>7s} {'s per seed, median':>19s} {'min':>9s} {'max':>9s} {'per epoch (50 seeds)':>21s}")
    for name, b, _ in cells:
        v = ts[(name, b)]; m = 50 * float(np.median(v))
        per = f"{m / 3600:.2f} h" if m >= 3600 else f"{m / 60:.1f} min"
        print(f"{name:6s} {b:7d} {float(np.median(v)):19.2f} {min(v):9.2f} {max(v):9.2f} {per:>21s}")
    for b in budgets:
        same = sum(x == y for x, y in zip(dec[("numpy", b)], dec[("jax", b)]))
        print(f"b{b}: reject decisions numpy {sum(dec[('numpy', b)])}/{len(seeds)}, jax {sum(dec[('jax', b)])}/{len(seeds)}, "
              f"same decision on {same} of {len(seeds)} seeds (the two implementations draw their own bootstrap streams)")
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
