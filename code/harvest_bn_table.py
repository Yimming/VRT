"""
harvest_bn_table.py — eval-mode versus train-mode BatchNorm table: sustained onset of VRT, MMD, MMDAgg, C2ST and ED under
eval-mode and train-mode extraction for every model with BatchNorm layers (ResNet20-FRN and the 5-layer CNN have none).
10,000 points per split (MMDAgg 5,000), seeds 42 to 91. Decision rules as Table 1: VRT, MMD and ED median p < 0.05; C2ST
median p < 0.05 and median held-out accuracy > 0.55; MMDAgg at least 60 percent of the seeds rejecting. Onset: first
evaluated epoch from which every later evaluated epoch rejects. Only epochs with all 50 seeds count.
Cells read eval -> train;  -  no sustained detection;  .  not computed yet (no complete epoch);  zoo rows min-max over five
models, with (k/5) when only k models detect;  Phase Zoo onsets at the first checkpoint print as <=100.

Usage:  python harvest_bn_table.py            table, plus the evaluated-epoch counts of the train-mode runs
        python harvest_bn_table.py -v         also every single model
        python harvest_bn_table.py --latex    rows for Table/table4.tex (cells not computed are left empty)
"""
import csv, glob, os, sys, collections
import numpy as np

BF = "/home3/p318770/venvs/overfitting/benchmark_final/s50"
SSLB = "/home3/p318770/venvs/overfitting/benchmark_final/ssl_baselines"   # eval-mode SimCLR MMDAgg 5,000
SEEDS = set(range(42, 92))
VERBOSE, LATEX = "-v" in sys.argv, "--latex" in sys.argv


def read(patterns, key, label, filt=lambda r: True, val="p", acc=False):
    """epoch -> {seed: p} (or (p, acc)) from every file matching the patterns, rows with r[key] == label"""
    d = collections.defaultdict(dict)
    for pat in patterns:
        for f in glob.glob(pat):
            if os.path.getsize(f) == 0:
                continue
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    try:
                        if r[key] != label or not filt(r) or int(r["seed"]) not in SEEDS:
                            continue
                        v = float(r[val]) if not acc else (float(r[val]), float(r["acc"]))
                        d[int(r["epoch"])].setdefault(int(r["seed"]), v)
                    except (KeyError, ValueError, TypeError):
                        continue
    return {e: v for e, v in d.items() if len(v) == len(SEEDS)}           # complete epochs only


def rejects(method, vals):
    v = list(vals.values())
    if method == "mmdagg":
        return np.mean(v) >= 0.6
    if method == "c2st":
        return np.median([p for p, _ in v]) < 0.05 and np.median([a for _, a in v]) > 0.55
    return np.median(v) < 0.05


def onset(method, d):
    if not d:
        return None, 0                                          # not computed
    eps = sorted(d); rej = [rejects(method, d[e]) for e in eps]
    o = next((e for i, e in enumerate(eps) if all(rej[i:])), "-")
    return o, len(eps)


def sources(tag, mode):
    """(method -> (patterns, key, label, filter, acc)) for one model and one extraction mode"""
    t = tag if mode == "eval" else f"{tag}_bntrain"
    if tag.startswith("simclr_"):
        ds = tag.split("_")[1]; lab = ds if mode == "eval" else f"{ds}_bntrain"
        # every group of ssl_s50_fill.py (fast, vrt, base, raw, extra, knn); read() keeps the first row per (epoch, seed)
        fast = [f"{BF}/ssl/ssl_{ds}_*_c*.csv"] if mode == "eval" else [f"{BF}/ssl_bntrain/ssl_{ds}_bntrain_*_c*.csv"]
        agg = [f"{SSLB}/sslagg_{ds}_*.csv"] if mode == "eval" else [f"{BF}/ssl_bntrain/sslagg_{ds}_bntrain_*.csv"]
        b10 = lambda s: (lambda r: r["budget"] == "b10000" and r["stat"] == s)
        return {"vrt": (fast, "model", lab, b10("vrt"), False), "mmd": (fast, "model", lab, b10("mmd"), False),
                "ed": (fast, "model", lab, b10("ed"), False), "c2st": (fast, "model", lab, b10("c2st"), True),
                "mmdagg": (agg, "model", lab, lambda r: r["budget"] == "b5000", False)}
    c2 = [f"{BF}/c2st/c2st_{t}_b*.csv"] + ([f"{BF}/bntrain_c2st/c2st_{t}_b*.csv"] if mode == "train" else [])
    ag = [f"{BF}/mmdagg/mmdagg_{t}_b5000*.csv"] + ([f"{BF}/bntrain_mmdagg/mmdagg_{t}_b5000*.csv"] if mode == "train" else [])
    return {"vrt": ([f"{BF}/vrt_fwd/vrt_fwd_{t}_large_c*.csv"], "tag", t, lambda r: True, False),
            "mmd": ([f"{BF}/mmd_p2000/mmd_{t}_n10000_c*.csv"], "tag", t, lambda r: True, False),
            "ed": ([f"{BF}/ed_p2000/ed_{t}_n10000_c*.csv"], "tag", t, lambda r: True, False),
            "c2st": (c2, "model", t, lambda r: r["budget"] == "b10000", True),
            "mmdagg": (ag, "model", t, lambda r: r["budget"] == "b5000", False)}


METHODS = [("VRT", "vrt"), ("MMD", "mmd"), ("MMDAgg", "mmdagg"), ("C2ST", "c2st"), ("ED", "ed")]
ROWS = [("ResNet-18 C100 Zoo ($\\times5$)", [f"zoo_c100_m{i}" for i in range(5)]),
        ("ResNet-18 C10 Zoo ($\\times5$)", [f"zoo_c10_m{i}" for i in range(5)]),
        ("ResNet-50 (CIFAR-100)", ["r50"]),
        ("Phase Zoo bs8", ["phase_bs8"]), ("Phase Zoo bs128", ["phase_bs128"]), ("Phase Zoo bs1{,}024", ["phase_bs1024"]),
        ("SimCLR CIFAR-10", ["simclr_cifar10"]), ("SimCLR CIFAR-100", ["simclr_cifar100"]),
        ("A: baseline", ["ablA"]), ("B: $10\\times$ wd", ["ablB"]), ("C: strong aug", ["ablC"]), ("D: wd + aug", ["ablD"]),
        ("DeepResNet50 simple (ImageNet)", ["imgnet_simple"]), ("DeepResNet50 adv.\\ (ImageNet)", ["imgnet_adv"])]

res = {}                                                         # (tag, mode, method) -> (onset, n complete epochs)
for _, tags in ROWS:
    for tag in tags:
        for mode in ("eval", "train"):
            src = sources(tag, mode)
            for _, m in METHODS:
                pats, key, lab, filt, acc = src[m]
                res[(tag, mode, m)] = onset(m, read(pats, key, lab, filt, acc=acc))


def show(tag, o):
    if o is None:
        return "."
    if tag.startswith("phase_") and o == 100:
        return "<=100"
    return str(o)


def group(tags, mode, m):
    os_ = [res[(t, mode, m)][0] for t in tags]
    if any(o is None for o in os_):
        return "."
    if len(tags) == 1:
        return show(tags[0], os_[0])
    hits = [o for o in os_ if o != "-"]
    if not hits:
        return "-"
    s = f"{min(hits)}-{max(hits)}" if min(hits) != max(hits) else f"{hits[0]}"
    return s if len(hits) == len(tags) else f"{s} ({len(hits)}/{len(tags)})"


w = max(len(n) for n, _ in ROWS)
print(f"{'model':{w}s}  " + "  ".join(f"{h:>18s}" for h, _ in METHODS))
latex = []
for name, tags in ROWS:
    cells = []
    for _, m in METHODS:
        e, t = group(tags, "eval", m), group(tags, "train", m)
        cells.append(f"{e} -> {t}")
    print(f"{name:{w}s}  " + "  ".join(f"{c:>18s}" for c in cells))
    if VERBOSE and len(tags) > 1:
        for tag in tags:
            print(f"{'   ' + tag:{w}s}  " + "  ".join(f"{show(tag, res[(tag, 'eval', m)][0]) + ' -> ' + show(tag, res[(tag, 'train', m)][0]):>18s}" for _, m in METHODS))
    if LATEX:
        def tex(s):
            return s.replace("<=", "$\\leq$").replace("-", "--") if s not in ("-", ".") else ("\\textemdash" if s == "-" else "")
        lc = []
        for _, m in METHODS:
            e, t = group(tags, "eval", m), group(tags, "train", m)
            if "." in (e, t) and not (e == "-" and t == "-"):
                lc.append("" if t == "." else f"{tex(e)} $\\rightarrow$ {tex(t)}")
            elif e == "-" and t == "-":
                lc.append("\\textemdash")
            else:
                lc.append(f"{tex(e)} $\\rightarrow$ {tex(t)}".replace("\\textemdash $", "\\textemdash\\ $"))
        latex.append(f"{name:36s} & " + " & ".join(lc) + " \\\\")

print("\ncomplete train-mode epochs per model (VRT MMD MMDAgg C2ST ED):")
for name, tags in ROWS:
    print(f"  {name:{w}s} " + "; ".join(f"{t}: " + " ".join(str(res[(t, 'train', m)][1]) for _, m in METHODS) for t in tags))
if LATEX:
    print("\n% rows for Table/table4.tex (eval -> train; empty = not computed yet)")
    print("\n".join(latex))
