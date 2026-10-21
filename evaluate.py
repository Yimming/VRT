"""Feature tests (Tables 1–3; Appendices B, E, G, I), geometry (J), and timing (K)."""
import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

from common import l2n, load_epoch, parse_range, validate_features
from vrt import compute_chat, perm_p, vrt_full, vrt_low, vrt_reversed

METHODS = ("vrt", "reversed", "mmd", "ed", "c2st", "mmdagg", "knn", "mmd-scores")
FIELDS = ("model", "epoch", "method", "budget", "features", "protocol", "seed",
          "n_perm", "backend", "statistic", "p", "accuracy", "decision")


def prepare_features(train, test, method, budget, protocol, raw=False):
    validate_features(train, test, allow_zero_rows=raw)
    if train.shape[1] != test.shape[1]:
        raise ValueError("Training and test features must have the same dimension")
    if min(len(train), len(test)) < budget:
        raise ValueError(f"Need at least {budget} points in each split")
    if raw and method in ("vrt", "reversed", "mmd-scores"):
        raise ValueError("VRT and its score transformation require normalized features")
    if raw:
        if protocol == "simclr" or method in ("c2st", "mmdagg"):
            return train.astype(np.float32), test.astype(np.float32)
        return train, test
    use64 = (method == "reversed" or
             method in ("vrt", "mmd-scores") and
             (protocol == "supervised" or budget == 1500))
    dtype = np.float64 if use64 else np.float32
    return l2n(train.astype(dtype)), l2n(test.astype(dtype))


def score_samples(train, test, seed, reference=2000, neighbours=60, queries=8000):
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(train))
    ref = train[order[:reference]]
    query = train[order[reference:reference + min(queries, len(train) - reference)]]
    return (compute_chat(query, ref, train.shape[1], neighbours),
            compute_chat(test, ref, train.shape[1], neighbours), rng)


def cap_infinite(a, b):
    pooled = np.concatenate([a, b])
    if np.isnan(pooled).any() or np.isneginf(pooled).any():
        raise ValueError("Scores contain NaN or negative infinity")
    if np.isposinf(pooled).any():
        finite = pooled[np.isfinite(pooled)]
        if not len(finite):
            raise ValueError("No finite scores for the MMD score transform")
        cap = float(finite.max() + 10.0 * (finite.max() - finite.min()))
        a, b = np.where(np.isposinf(a), cap, a), np.where(np.isposinf(b), cap, b)
    return a, b


def run_test(train, test, method="vrt", budget=10000, seed=42, n_perm=2000,
             protocol="supervised", raw=False, backend="numpy"):
    """Return records for one seed, preserving the paper's per-method preprocessing."""
    import baselines as base
    x, y = prepare_features(train, test, method, budget, protocol, raw)
    d = x.shape[1]
    if method == "vrt":
        if budget == 10000:
            if len(y) != 10000:
                raise ValueError("The main VRT protocol requires exactly 10,000 test rows")
            statistic, p = vrt_full(x, y, d, n_perm=n_perm, seed=seed)
        elif budget == 1500:
            statistic, p = vrt_low(x, y, d, n_perm=n_perm, seed=seed)
        else:
            raise ValueError("VRT budgets are 10,000 or 1,500; use synthetic.py for custom sizes")
    elif method == "reversed":
        if budget not in (10000, 1500):
            raise ValueError("Reversed VRT budgets are 10,000 or 1,500")
        reference, neighbours, queries = (2000, 60, 8000) if budget == 10000 else (500, 15, 1500)
        rng = np.random.RandomState(1000 + seed)
        a = x[rng.permutation(len(x))[:budget]]
        b = y[rng.permutation(len(y))[:budget]]
        statistic, p = vrt_reversed(a, b, d, reference, neighbours, queries, n_perm, seed)
    elif method == "mmd":
        statistic, p = base.mmd_test_fast(x, y, budget, n_perm, seed, backend)
    elif method == "ed":
        statistic, p = base.energy_test_fast(x, y, budget, n_perm, seed, backend)
    elif method == "c2st":
        accuracy, p = base.c2st_test(x, y, n_sub=budget, seed=seed)
        return [dict(method=method, statistic="", p=p, accuracy=accuracy, decision="")]
    elif method == "mmdagg":
        decision = base.mmdagg_test(x, y, n_sub=budget, seed=seed)
        return [dict(method=method, statistic="", p="", accuracy="", decision=decision)]
    elif method == "knn":
        rng = np.random.default_rng(seed)
        a = x[rng.choice(len(x), budget, replace=False)]
        b = y[rng.choice(len(y), budget, replace=False)]
        labels = np.array([0] * budget + [1] * budget)
        graph = base.knn_graph(np.vstack([a, b]), 3, "torch" if backend == "torch" else "sklearn")
        ps = base.schilling_p(graph, labels, budget, budget, 3)
        ph = base.henze_p(graph, labels, n_perm, seed)
        return [dict(method=m, statistic="", p=p, accuracy="", decision="")
                for m, p in (("schilling", ps), ("henze", ph))]
    elif method == "mmd-scores":
        if budget != 10000:
            raise ValueError("Appendix I uses the main 10,000-point protocol")
        a, b, _ = score_samples(x, y, seed)
        if protocol == "simclr":
            before = float(np.median(b) - np.median(a))
            a, b = cap_infinite(a, b)
            if float(np.median(b) - np.median(a)) != before:
                raise ValueError("Capping infinite scores changed the median difference")
        statistic, p = base.mmd_test_fast(a[:, None], b[:, None], budget, n_perm, seed, backend)
    else:
        raise ValueError(f"Unknown method: {method}")
    if not (np.isfinite(statistic) and np.isfinite(p) and 0 <= p <= 1):
        raise FloatingPointError(f"Invalid {method} result")
    return [dict(method=method, statistic=statistic, p=p, accuracy="", decision="")]


QS = (0.10, 0.25, 0.75, 0.90)
STATS = ("med", "mean", "tmean", "q10", "q25", "q75", "q90", "sd", "ks")

def summaries(c):
    q = np.quantile(c, QS)
    return {'median': float(np.median(c)), 'mean': float(c.mean()), 'sd': float(c.std()), 'q10': float(q[0]), 'q25': float(q[1]), 'q75': float(q[2]), 'q90': float(q[3])}

def stat_vector(a, b):
    sa, sb = (np.sort(a), np.sort(b))
    na, nb = (len(sa), len(sb))
    ta, tb = (int(0.1 * na), int(0.1 * nb))
    qa, qb = (np.quantile(sa, QS), np.quantile(sb, QS))
    allv = np.concatenate([sa, sb])
    fa = np.searchsorted(sa, allv, side='right') / na
    fb = np.searchsorted(sb, allv, side='right') / nb
    ks = float(np.max(np.abs(fb - fa)))
    return np.array([np.median(sb) - np.median(sa), sb.mean() - sa.mean(), sb[tb:nb - tb].mean() - sa[ta:na - ta].mean(), qb[0] - qa[0], qb[1] - qa[1], qb[2] - qa[2], qb[3] - qa[3], sb.std() - sa.std(), ks])

def centroid_stats(e_tr, e_te, y_tr, y_te, n_classes):
    thetas, dRs = ([], [])
    for c in range(n_classes):
        m_tr = e_tr[y_tr == c].mean(axis=0)
        m_te = e_te[y_te == c].mean(axis=0)
        cos = np.dot(m_tr, m_te) / (np.linalg.norm(m_tr) * np.linalg.norm(m_te))
        thetas.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        dRs.append(np.linalg.norm(m_tr) - np.linalg.norm(m_te))
    g_tr = e_tr.mean(axis=0)
    g_te = e_te.mean(axis=0)
    cosg = np.dot(g_tr, g_te) / (np.linalg.norm(g_tr) * np.linalg.norm(g_te))
    theta_glob = np.degrees(np.arccos(np.clip(cosg, -1, 1)))
    dR_glob = np.linalg.norm(g_tr) - np.linalg.norm(g_te)
    return (float(np.mean(thetas)), theta_glob, float(np.mean(dRs)), float(dR_glob))

def split_half_null(e_tr, y_tr, y_te, n_classes, n_rep=5, seed=0):
    rng = np.random.RandomState(seed)
    te_counts = np.bincount(y_te, minlength=n_classes)
    reps = []
    for _ in range(n_rep):
        thetas = []
        for c in range(n_classes):
            idx = np.where(y_tr == c)[0]
            idx = idx[rng.permutation(len(idx))]
            k = te_counts[c]
            m_a = e_tr[idx[:k]].mean(axis=0)
            m_b = e_tr[idx[k:]].mean(axis=0)
            cos = np.dot(m_a, m_b) / (np.linalg.norm(m_a) * np.linalg.norm(m_b))
            thetas.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        reps.append(np.mean(thetas))
    return (float(np.mean(reps)), float(np.std(reps)))

def new_csv(path, fieldnames):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = target.open("x", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return handle, writer


def checkpoint_test(args):
    methods = args.methods.split(",")
    if any(m not in METHODS for m in methods):
        raise ValueError(f"Methods must be chosen from {METHODS}")
    handle, writer = new_csv(args.out, FIELDS)
    evaluated = 0
    with handle:
        for epoch in parse_range(args.epochs):
            loaded = load_epoch(args.features, epoch)
            if loaded is None:
                raise FileNotFoundError(args.features.format(ep=epoch))
            for method in methods:
                budget = 5000 if method == "mmdagg" and args.budget == 10000 else args.budget
                for seed in parse_range(args.seeds):
                    for result in run_test(*loaded, method, budget, seed, args.permutations,
                                           args.protocol, args.raw, args.backend):
                        writer.writerow(dict(model=args.model, epoch=epoch, budget=budget,
                            features="raw" if args.raw else "l2", protocol=args.protocol, seed=seed,
                            n_perm=args.permutations, backend=args.backend, **result))
                handle.flush()
                print(f"epoch={epoch} method={method} budget={budget}", flush=True)
                evaluated += 1
    print(f"Wrote {evaluated} checkpoint/method comparisons to {args.out}")


def timing(args):
    from threadpoolctl import threadpool_limits
    loaded = load_epoch(args.features, args.epoch)
    if loaded is None:
        raise FileNotFoundError(args.features.format(ep=args.epoch))
    x, y = prepare_features(*loaded, "vrt", args.budget, "supervised")
    fields = ("model", "epoch", "budget", "seed", "threads", "seconds", "statistic", "p")
    handle, writer = new_csv(args.out, fields)
    seconds = []
    with handle, threadpool_limits(limits=args.threads):
        fn = vrt_full if args.budget == 10000 else vrt_low
        if args.budget == 10000 and len(y) != 10000:
            raise ValueError("Main VRT timing needs exactly 10,000 test rows")
        fn(x, y, x.shape[1], seed=41)
        for seed in parse_range(args.seeds):
            start = time.perf_counter()
            statistic, p = fn(x, y, x.shape[1], seed=seed)
            duration = time.perf_counter() - start
            seconds.append(duration)
            writer.writerow(dict(model=args.model, epoch=args.epoch, budget=args.budget,
                seed=seed, threads=args.threads, seconds=duration, statistic=statistic, p=p))
    print(f"CPU host={platform.node()}, threads={args.threads}, median={np.median(seconds):.4f}s")


def smoke(args):
    """Exercise numerical paths on synthetic inputs; these are not power estimates."""
    import baselines as base
    rng = np.random.RandomState(700)
    x = rng.standard_normal((1500, 16))
    y = rng.standard_normal((1500, 16)) + .05
    report = []
    for method in ("vrt", "reversed", "mmd", "ed", "c2st", "knn"):
        n = 1500 if method in ("vrt", "reversed") else 80
        rows = run_test(x, y, method, n, 42, 19, backend=args.backend)
        report.extend(rows)
    if args.include_mmdagg:
        report.extend(run_test(x, y, "mmdagg", 40, 42, 19))
    a, b = l2n(x), l2n(y)
    statistic, p = vrt_full(a, b, 16, M=100, k=5, n_qtr=200, n_perm=19)
    report.append(dict(method="vrt-full-custom", statistic=statistic, p=p))
    scores_a, scores_b, _ = score_samples(a, b, 42, reference=100, neighbours=5, queries=200)
    statistic, p = base.mmd_test_fast(scores_a[:, None], scores_b[:, None], 80, 19, 42, args.backend)
    report.append(dict(method="mmd-scores-custom", statistic=statistic, p=p))
    print(json.dumps(report, indent=2))
    print("SMOKE PASSED")


def score_statistics_test(a, b, rng, n_perm):
    """Appendix I: nine statistics using the VRT permutation stream."""
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("The mean and quantile analyses require finite scores")
    observed = stat_vector(a, b)
    pooled = np.concatenate([a, b])
    count = np.zeros(len(STATS))
    for _ in range(n_perm):
        order = rng.permutation(len(pooled))
        value = stat_vector(pooled[order[:len(a)]], pooled[order[len(a):]])
        count[:-1] += np.abs(value[:-1]) >= np.abs(observed[:-1])
        count[-1] += value[-1] >= observed[-1]
    return observed, (count + 1) / (n_perm + 1)


def score_statistics(args):
    handle, writer = new_csv(args.out, ("tag", "epoch", "seed", "stat", "obs", "p"))
    saved = {}
    with handle:
        for epoch in parse_range(args.epochs):
            loaded = load_epoch(args.features, epoch)
            if loaded is None:
                raise FileNotFoundError(args.features.format(ep=epoch))
            x, y = prepare_features(*loaded, "vrt", 10000, "supervised")
            for seed in parse_range(args.seeds):
                a, b, rng = score_samples(x, y, seed)
                observed, p = score_statistics_test(a, b, rng, args.permutations)
                for name, value, pv in zip(STATS, observed, p):
                    writer.writerow(dict(tag=args.model, epoch=epoch, seed=seed,
                                         stat=name, obs=value, p=pv))
                for side, scores in (("tr", a), ("te", b)):
                    for name, value in summaries(scores).items():
                        writer.writerow(dict(tag=args.model, epoch=epoch, seed=seed,
                                             stat=f"{side}_{name}", obs=value, p=""))
                if seed == args.check_seed:
                    saved[f"ep{epoch:03d}_tr"] = a
                    saved[f"ep{epoch:03d}_te"] = b
            handle.flush()
            print(f"epoch={epoch}: score statistics written", flush=True)
    if saved:
        score_path = Path(args.out).with_suffix(f".scores_s{args.check_seed}.npz")
        with score_path.open("xb") as target:
            np.savez(target, **saved)


def geometry(args):
    with np.load(args.labels) as labels:
        y_tr, y_te = labels["y_tr"], labels["y_te"]
    classes = np.unique(np.concatenate([y_tr, y_te]))
    n_classes = len(classes)
    if (y_tr.ndim != 1 or y_te.ndim != 1 or
        not np.array_equal(classes, np.arange(n_classes)) or
        not np.issubdtype(y_tr.dtype, np.integer) or
        not np.issubdtype(y_te.dtype, np.integer)):
        raise ValueError("Labels must be integer vectors with classes 0, ..., C-1")
    for c in classes:
        if not 0 < (y_te == c).sum() < (y_tr == c).sum():
            raise ValueError("Each class needs more training than test examples for the disjoint control")
    fields = ("model", "epoch", "theta_bar_deg", "theta_glob_deg", "theta_null_deg",
              "theta_null_std", "dR_bar", "dR_glob")
    handle, writer = new_csv(args.out, fields)
    with handle:
        for epoch in parse_range(args.epochs):
            loaded = load_epoch(args.features, epoch)
            if loaded is None:
                raise FileNotFoundError(args.features.format(ep=epoch))
            validate_features(*loaded)
            x, y = (l2n(v.astype(np.float64)) for v in loaded)
            if len(x) != len(y_tr) or len(y) != len(y_te):
                raise ValueError("Label arrays must match the feature rows, in the same order")
            tb, tg, rb, rg = centroid_stats(x, y, y_tr, y_te, n_classes)
            tn, tns = split_half_null(x, y_tr, y_te, n_classes, seed=epoch)
            values = (tb, tg, tn, tns, rb, rg)
            if not np.isfinite(values).all():
                raise ValueError("Undefined centroid angle: check class means and feature values")
            writer.writerow(dict(zip(fields, (args.model, epoch, *values))))
            handle.flush()
            print(f"epoch={epoch}: class angle={tb:.2f}, global angle={tg:.2f}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("test", help="Tables 1–3; Appendices B/E/G/I: run on checkpoint features")
    run.add_argument("--features", required=True, help="NPZ path or pattern such as features/ep{ep:03d}.npz")
    run.add_argument("--model", required=True)
    run.add_argument("--epochs", required=True)
    run.add_argument("--methods", default="vrt", help=",".join(METHODS))
    run.add_argument("--budget", type=int, choices=(1500, 10000), default=10000)
    run.add_argument("--seeds", default="42-91")
    run.add_argument("--permutations", type=int, default=2000)
    run.add_argument("--protocol", choices=("supervised", "simclr"), default="supervised")
    run.add_argument("--raw", action="store_true")
    run.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    run.add_argument("--out", required=True)
    run.set_defaults(func=checkpoint_test)
    bench = commands.add_parser("timing", help="Appendix K: CPU VRT timing, excluding feature extraction")
    bench.add_argument("--features", required=True)
    bench.add_argument("--model", required=True)
    bench.add_argument("--epoch", type=int, required=True)
    bench.add_argument("--budget", type=int, choices=(1500, 10000), default=10000)
    bench.add_argument("--seeds", default="42-51")
    bench.add_argument("--threads", type=int, default=8)
    bench.add_argument("--out", required=True)
    bench.set_defaults(func=timing)
    stats = commands.add_parser("score-statistics", help="Appendix I: median, mean, quantile and spread tests")
    stats.add_argument("--features", required=True)
    stats.add_argument("--model", required=True)
    stats.add_argument("--epochs", required=True)
    stats.add_argument("--seeds", default="42-91")
    stats.add_argument("--permutations", type=int, default=2000)
    stats.add_argument("--check-seed", type=int, default=42)
    stats.add_argument("--out", required=True)
    stats.set_defaults(func=score_statistics)
    geom = commands.add_parser("geometry", help="Appendix J: class-centroid angles and concentration gaps")
    geom.add_argument("--features", required=True)
    geom.add_argument("--labels", required=True, help="NPZ with y_tr/y_te in the same row order as features")
    geom.add_argument("--model", required=True)
    geom.add_argument("--epochs", required=True)
    geom.add_argument("--out", required=True)
    geom.set_defaults(func=geometry)
    check = commands.add_parser("smoke", help="Small CPU/GPU execution test; no real data needed")
    check.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    check.add_argument("--include-mmdagg", action="store_true")
    check.set_defaults(func=smoke)
    args = parser.parse_args(argv)
    if getattr(args, "permutations", 1) < 1:
        parser.error("--permutations must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
