"""
find_grid_gaps.py — which onset cells of the camera-ready are still on a coarse epoch grid, and the task lines that make
them exact (October 10).

For every cell (model tag, extraction mode, statistic, budget) the script loads the per-epoch results with the same file
patterns as harvest_s50.py / harvest_bn_table.py, keeps the epochs with all 50 seeds, applies the paper's decision rule
(median p < 0.05; C2ST also median accuracy > 0.55; MMDAgg at least 60 percent of the seeds rejecting) and finds the
sustained onset e1. Let e0 be the last evaluated epoch before e1 (it does not reject, by definition of e1). The cell is
exact when every saved checkpoint strictly between e0 and e1 was evaluated; otherwise those checkpoints are the gap and
are printed, together with a task line (ssl_s50_fill.py / base_fill.py / ssl_mmdagg5k_seeds.py formats) that fills it.
Null cells are exact when the last saved checkpoint was evaluated (reported otherwise). Saved checkpoints = the epochs of
the eval-mode 10,000-point VRT cell of the model (1 to 1,000 for SimCLR).

Usage (cluster, code folder, venv active):  python find_grid_gaps.py            prints gaps and task lines
                                            python find_grid_gaps.py --tasks     writes gap_tasks_<group>.txt files
                                            python find_grid_gaps.py --tails     also lists, per detecting cell, the saved
                                                                                 checkpoints after the onset that were not
                                                                                 evaluated (the sustained claim covers only
                                                                                 evaluated checkpoints there)
"""
import csv, glob, io, os, sys, collections
import numpy as np

BF = os.environ.get("S50", "/home3/p318770/venvs/overfitting/benchmark_final/s50")
SSLB = os.environ.get("SSLB", "/home3/p318770/venvs/overfitting/benchmark_final/ssl_baselines")
CODE = "/scratch/p318770/ssl_supcon_code_s50"
FEATS = "/scratch/p318770/model_zoos/ladder_feats"
NS = 50
CIFAR = ["v7", "r50", "ablA", "ablB", "ablC", "ablD"] + [f"zoo_c10_m{i}" for i in range(5)] + [f"zoo_c100_m{i}" for i in range(5)]
OTHER = ["phase_bs8", "phase_bs128", "phase_bs1024", "wp", "imgnet_simple", "imgnet_adv"]
SSL = {"simclr_c10": "cifar10", "simclr_c100": "cifar100"}
WRITE = "--tasks" in sys.argv
TAILS = "--tails" in sys.argv


_FILES = {}


def parse(f, pcol, acccol):
    """one file, parsed once: (model, stat, budget) -> epoch -> seed -> (p, acc); files without those columns key on None"""
    key = (f, pcol, acccol)
    if key in _FILES:
        return _FILES[key]
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    raw = open(f, "rb").read().replace(b"\x00", b"")
    for r in csv.DictReader(io.StringIO(raw.decode())):
        try:
            k = (r.get("model"), r.get("stat"), r.get("budget"))
            acc = float(r[acccol]) if acccol else float("nan")
            out[k][int(r["epoch"])].setdefault(int(r["seed"]), (float(r[pcol]), acc))
        except (KeyError, ValueError):
            continue
    _FILES[key] = out
    return out


def read(patterns, pcol="p", acccol=None, filt=None):
    """epoch -> {seed: (p, acc)} over the files of the patterns, rows matching filt (model / stat / budget)"""
    d = collections.defaultdict(dict)
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            for (model, stat, budget), eps in parse(f, pcol, acccol).items():
                if filt and any({"model": model, "stat": stat, "budget": budget}.get(k) != v for k, v in filt.items()):
                    continue
                for e, seeds in eps.items():
                    for sd, v in seeds.items():
                        d[e].setdefault(sd, v)
    return d


def cell(tag, mode, stat, N):
    """epoch -> {seed: (p, acc)}; mode eval or train"""
    if tag in SSL:
        ds = SSL[tag]
        lab = ds if mode == "eval" else f"{ds}_bntrain"
        if stat == "mmdagg" and N == 5000:
            pats = [f"{SSLB}/sslagg_{ds}_*.csv"] if mode == "eval" else [f"{BF}/ssl_bntrain/sslagg_{ds}_bntrain_*.csv"]
            return read(pats, filt={"model": lab, "budget": "b5000", "stat": "mmdagg"})
        st = {"vrt": "vrt" if N == 10000 else "vrt_low"}.get(stat, stat)
        pats = [f"{BF}/ssl/ssl_{ds}_*_c*.csv"] if mode == "eval" else [f"{BF}/ssl_bntrain/ssl_{ds}_bntrain_*_c*.csv"]
        return read(pats, "p", "acc", filt={"model": lab, "stat": st, "budget": f"b{N}"})
    t = tag if mode == "eval" else f"{tag}_bntrain"
    b = "large" if N == 10000 else "small"
    if stat == "vrt":
        return read([f"{BF}/vrt_fwd/vrt_fwd_{t}_{b}_c*.csv"])
    if stat == "vrt_rev":
        return read([f"{BF}/reversed/vrt_rev_{t}_{b}_c*.csv"])
    if stat in ("mmd", "ed", "mmd_raw", "ed_raw"):
        base = {"mmd": "mmd_p2000", "ed": "ed_p2000", "mmd_raw": "mmd_raw_p2000", "ed_raw": "ed_raw_p2000"}[stat]
        return read([f"{BF}/{base}/{stat.split('_')[0]}_{t}_n{N}_c*.csv"])
    if stat in ("henze", "schilling"):
        return read([f"{BF}/knn_v2/knn_{t}_n{N}_part*.csv"], f"p_{stat}")
    if stat == "c2st":
        pats = [f"{BF}/c2st/c2st_{t}_b*_c*.csv"] + ([f"{BF}/bntrain_c2st/c2st_{t}_b*_c*.csv"] if mode == "train" else [])
        return read(pats, "p", "acc", filt={"model": t, "budget": f"b{N}"})
    if stat == "mmdagg":
        pats = [f"{BF}/mmdagg/mmdagg_{t}_b{N}_c*.csv"] + ([f"{BF}/bntrain_mmdagg/mmdagg_{t}_b{N}_c*.csv"] if mode == "train" else [])
        return read(pats)
    raise ValueError(stat)


def rejects(stat, vals):
    ps = np.array([p for p, _ in vals]); accs = np.array([a for _, a in vals])
    if stat == "mmdagg":
        return ps.mean() >= 0.6
    if stat == "c2st":
        return np.median(ps) < 0.05 and np.median(accs) > 0.55
    return np.median(ps) < 0.05


def checkpoints(tag):
    if tag in SSL:
        return list(range(1, 1001))
    d = cell(tag, "eval", "vrt", 10000)
    return sorted(d)


STATS_LARGE = [("vrt", 10000), ("mmd", 10000), ("mmdagg", 5000), ("c2st", 10000), ("ed", 10000),
               ("mmd_raw", 10000), ("ed_raw", 10000), ("henze", 10000), ("schilling", 10000), ("vrt_rev", 10000)]
STATS_SMALL = [("vrt", 1500), ("mmd", 1500), ("mmdagg", 1500), ("c2st", 1500), ("ed", 1500),
               ("henze", 1500), ("schilling", 1500), ("vrt_rev", 1500)]


def task_line(tag, mode, stat, N, lo, hi):
    """one task line that evaluates epochs lo..hi of this cell; None when the format is not known here"""
    if tag in SSL:
        ds = SSL[tag]; ck = f'"$(ls -d /scratch/p318770/model_zoos/ssl_supcon/SimCLR_{ds}_* | head -1)"'
        if stat == "mmdagg" and N == 5000:
            if mode == "eval":
                return None
            eps = ",".join(str(e) for e in range(lo, hi + 1))
            return (f"cd {CODE} && python ssl_mmdagg5k_seeds.py --ckpt_dir {ck} --dataset {ds} --bn_train --epochs {eps} "
                    f"--seeds 42-91 --out_dir {BF}/ssl_bntrain")
        group = {"vrt": "vrt", "mmd": "base", "ed": "base", "c2st": "base", "mmd_raw": "raw", "ed_raw": "raw",
                 "mmdagg": "slow", "henze": "knn", "schilling": "knn", "vrt_rev": "rev"}[stat]
        if mode == "eval" and group in ("knn", "rev"):
            group = "extra"
        out = f"{BF}/ssl" if mode == "eval" else f"{BF}/ssl_bntrain"
        bn = " --bn_train" if mode == "train" else ""
        return (f"cd {CODE} && python ssl_s50_fill.py --ckpt_dir {ck} --dataset {ds} --group {group}{bn} --every 1 "
                f"--lo {lo} --hi {hi} --seeds 42-91 --out_dir {out}")
    t = tag if mode == "eval" else f"{tag}_bntrain"
    if stat == "mmdagg":
        out = f"{BF}/mmdagg" if mode == "eval" else f"{BF}/bntrain_mmdagg"
        return (f"cd {CODE} && python base_fill.py --stat mmdagg --tag {t} --npz_fmt '{FEATS}/{t}/ep{{ep:03d}}.npz' "
                f"--budgets {N} --every 1 --seeds 42-91 --epochs {lo}-{hi} --out_dir {out} --out_suffix _c{lo}-{hi}")
    if stat == "c2st":
        out = f"{BF}/c2st" if mode == "eval" else f"{BF}/bntrain_c2st"
        return (f"cd {CODE} && python base_fill.py --stat c2st --tag {t} --npz_fmt '{FEATS}/{t}/ep{{ep:03d}}.npz' "
                f"--budgets {N} --every 1 --seeds 42-91 --epochs {lo}-{hi} --out_dir {out} --out_suffix _c{lo}-{hi}")
    return None


def main():
    tasks = collections.defaultdict(list)
    n_cells = n_gap = n_tail = 0
    for tag in CIFAR + OTHER + list(SSL):
        cps = checkpoints(tag)
        if not cps:
            print(f"{tag:14s} no VRT checkpoints found, skipped"); continue
        for mode in ("eval", "train"):
            for stat, N in STATS_LARGE + STATS_SMALL:
                try:
                    d = cell(tag, mode, stat, N)
                except ValueError:
                    continue
                comp = sorted(e for e in d if len(d[e]) >= NS)
                if not comp:
                    continue
                n_cells += 1
                rej = {e: rejects(stat, list(d[e].values())) for e in comp}
                e1 = next((e for i, e in enumerate(comp) if all(rej[x] for x in comp[i:])), None)
                if e1 is None:
                    if cps[-1] not in comp:
                        print(f"{tag:14s} {mode:5s} {stat:9s} {N:5d}  null, but last checkpoint {cps[-1]} not evaluated (last evaluated {comp[-1]})")
                    continue
                earlier = [e for e in comp if e < e1]
                e0 = earlier[-1] if earlier else None
                gap = [c for c in cps if (e0 is None or c > e0) and c < e1 and c not in comp]
                tail = [c for c in cps if c > e1 and c not in comp]
                if tail and TAILS:
                    n_tail += 1
                    print(f"{tag:14s} {mode:5s} {stat:9s} {N:5d}  onset {e1}; unevaluated after the onset: {len(tail)} of {sum(c > e1 for c in cps)} checkpoints ({tail[0]} to {tail[-1]})")
                if not gap:
                    continue
                n_gap += 1
                print(f"{tag:14s} {mode:5s} {stat:9s} {N:5d}  onset {e1} (previous evaluated {e0}); unevaluated {gap[0]} to {gap[-1]} ({len(gap)})")
                line = task_line(tag, mode, stat, N, gap[0], gap[-1])
                key = ("ssl_" + mode if tag in SSL else "cifar_" + mode) + "_" + ("mmdagg" if stat == "mmdagg" else "other")
                tasks[key].append(line or f"# MANUAL  {tag} {mode} {stat} {N} epochs {gap[0]}-{gap[-1]}")
    print(f"\n{n_cells} cells checked, {n_gap} with a grid gap before the onset" + (f", {n_tail} with unevaluated checkpoints after the onset" if TAILS else ""))
    for key, lines in tasks.items():
        uniq = list(dict.fromkeys(lines))
        print(f"\n## {key}: {len(uniq)} task lines")
        for l in uniq:
            print(l)
        if WRITE:
            with open(f"gap_tasks_{key}.txt", "w") as f:
                f.write("\n".join(l for l in uniq if not l.startswith("#")) + "\n")
            print(f"written gap_tasks_{key}.txt")


if __name__ == "__main__":
    main()
