"""
harvest_raw_table.py — Table 3 of the camera-ready: sustained onset of MMD and ED on UNNORMALIZED features, eval-mode and
train-mode BatchNorm extraction, 10,000 points per split, 2,000 permutations, seeds 42 to 91 (median p < 0.05; onset = first
evaluated epoch from which every later evaluated epoch rejects; only epochs with all 50 seeds count).
Reads s50/mmd_raw_p2000/mmd_<tag>[_bntrain]_n10000_c*.csv and s50/ed_raw_p2000/ed_<tag>[_bntrain]_n10000_c*.csv; for the
SimCLR models the stats mmd_raw and ed_raw (budget b10000) of ssl_s50_fill.py, eval mode in s50/ssl/ssl_<ds>_*_c*.csv and
train mode in s50/ssl_bntrain/ssl_<ds>_bntrain_*_c*.csv.
Cells read eval -> train for the models with BatchNorm layers (ResNet20-FRN and the 5-layer CNN print the eval value only);
-  no sustained detection;  .  not computed yet;  zoo rows min-max over the five models with (k) when only k detect;
Phase Zoo onsets at the first checkpoint print as <=100.
Usage:  python harvest_raw_table.py            table plus the evaluated-epoch counts per model and mode
        python harvest_raw_table.py --latex    rows for Table/table3.tex
"""
import csv, glob, os, sys, collections
import numpy as np

BF = os.environ.get("S50_BF", "/home3/p318770/venvs/overfitting/benchmark_final/s50")
SEEDS = set(range(42, 92))
LATEX = "--latex" in sys.argv
NO_BN = {"v7", "wp"}


def read(pattern, tag):
    d = collections.defaultdict(dict)
    for f in glob.glob(pattern):
        if os.path.getsize(f) == 0:
            continue
        with open(f) as fh:
            for r in csv.DictReader(fh):
                try:
                    if r["tag"] != tag or int(r["seed"]) not in SEEDS:
                        continue
                    d[int(r["epoch"])].setdefault(int(r["seed"]), float(r["p"]))
                except (KeyError, ValueError, TypeError):
                    continue
    return {e: v for e, v in d.items() if len(v) == len(SEEDS)}


def read_ssl(pattern, model, stat):
    """epoch -> {seed: p} from ssl_s50_fill.py files (model,epoch,budget,stat,seed,acc,p), budget b10000"""
    d = collections.defaultdict(dict)
    for f in glob.glob(pattern):
        if os.path.getsize(f) == 0:
            continue
        with open(f) as fh:
            for r in csv.DictReader(fh):
                try:
                    if r["model"] != model or r["stat"] != stat or r["budget"] != "b10000" or int(r["seed"]) not in SEEDS:
                        continue
                    d[int(r["epoch"])].setdefault(int(r["seed"]), float(r["p"]))
                except (KeyError, ValueError, TypeError):
                    continue
    return {e: v for e, v in d.items() if len(v) == len(SEEDS)}


def onset(d):
    if not d:
        return None, 0
    eps = sorted(d)
    rej = [np.median(list(d[e].values())) < 0.05 for e in eps]
    return next((e for i, e in enumerate(eps) if all(rej[i:])), "-"), len(eps)


ROWS = [("ResNet20-FRN (CIFAR-10)", ["v7"]),
        ("ResNet-18 C100 Zoo ($\\times5$)", [f"zoo_c100_m{i}" for i in range(5)]),
        ("ResNet-18 C10 Zoo ($\\times5$)", [f"zoo_c10_m{i}" for i in range(5)]),
        ("ResNet-50 (CIFAR-100)", ["r50"]),
        ("Phase Zoo bs8", ["phase_bs8"]), ("Phase Zoo bs128", ["phase_bs128"]), ("Phase Zoo bs1{,}024", ["phase_bs1024"]),
        ("SimCLR CIFAR-10", ["simclr_c10"]), ("SimCLR CIFAR-100", ["simclr_c100"]),
        ("A: baseline", ["ablA"]), ("B: $10\\times$ wd", ["ablB"]), ("C: strong aug", ["ablC"]), ("D: wd + aug", ["ablD"]),
        ("5-layer CNN (CIFAR-10)", ["wp"]),
        ("DeepResNet50 simple (ImageNet)", ["imgnet_simple"]), ("DeepResNet50 adv.\\ (ImageNet)", ["imgnet_adv"])]
METHODS = [("MMD", "mmd_raw_p2000", "mmd"), ("ED", "ed_raw_p2000", "ed")]

res = {}
for _, tags in ROWS:
    for tag in tags:
        for mode in ("eval", "train"):
            t = tag if mode == "eval" else f"{tag}_bntrain"
            for name, folder, pref in METHODS:
                if tag.startswith("simclr_"):
                    ds = {"simclr_c10": "cifar10", "simclr_c100": "cifar100"}[tag]
                    model = ds if mode == "eval" else f"{ds}_bntrain"
                    sub = "ssl" if mode == "eval" else "ssl_bntrain"
                    res[(tag, mode, name)] = onset(read_ssl(f"{BF}/{sub}/ssl_{model}_*_c*.csv", model, f"{pref}_raw"))
                else:
                    res[(tag, mode, name)] = onset(read(f"{BF}/{folder}/{pref}_{t}_n10000_c*.csv", t))


def show(tag, o):
    if o is None:
        return "."
    if tag.startswith("phase_") and o == 100:
        return "<=100"
    return str(o)


def group(tags, mode, name):
    os_ = [res[(t, mode, name)][0] for t in tags]
    if any(o is None for o in os_):
        return "."
    if len(tags) == 1:
        return show(tags[0], os_[0])
    hits = [o for o in os_ if o != "-"]
    if not hits:
        return "-"
    s = f"{min(hits)}-{max(hits)}" if min(hits) != max(hits) else f"{hits[0]}"
    return s if len(hits) == len(tags) else f"{s} ({len(hits)})"


def cell(tags, name):
    e, t = group(tags, "eval", name), group(tags, "train", name)
    if tags[0] in NO_BN:
        return e
    return f"{e} -> {t}"


def tex(s):
    if s == ".":
        return ""
    if s == "-":
        return "\\textemdash"
    return s.replace("<=", "$\\leq$").replace("-", "--")


def tex_cell(tags, name):
    e, t = group(tags, "eval", name), group(tags, "train", name)
    if tags[0] in NO_BN or (e == "-" and t == "-"):
        return tex(e)
    if t == ".":
        return tex(e)                                   # train mode not computed yet: eval value only
    left = tex(e) + ("\\ " if e == "-" else "")
    return f"{left} $\\rightarrow$ {tex(t)}"


w = max(len(n) for n, _ in ROWS)
print(f"{'model':{w}s}  {'MMD raw':>22s}  {'ED raw':>22s}")
latex = []
for name, tags in ROWS:
    print(f"{name:{w}s}  " + "  ".join(f"{cell(tags, m):>22s}" for m, _, _ in METHODS))
    latex.append(f"{name:36s} & " + " & ".join(tex_cell(tags, m) for m, _, _ in METHODS) + " \\\\")
print("\ncomplete epochs per model, eval / train (MMD raw, ED raw):")
for name, tags in ROWS:
    print(f"  {name:{w}s} " + "; ".join(f"{t}: " + " ".join(f"{res[(t, 'eval', m)][1]}/{res[(t, 'train', m)][1]}" for m, _, _ in METHODS) for t in tags))
if LATEX:
    print("\n% rows for Table/table3.tex (eval -> train; eval value only where train mode is not computed yet)")
    print("\n".join(latex))
