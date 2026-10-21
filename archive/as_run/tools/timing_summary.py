"""
timing_summary.py — wall-clock cost of the tests, read from the SLURM logs of the 50-seed runs (camera-ready #27102,
October 15). Nothing is recomputed; the script only parses the per-epoch timing lines the run scripts printed.

Log families and what their timing lines cover
  ssl_s50_fill.py   (SimCLR, GPU or CPU)   "ep N  <stats>  (feat F s, total T s)": one checkpoint, every statistic of the
                                            --group (vrt: VRT b10000 + VRT b1500; base: MMD, ED, C2ST at b1500 and b10000;
                                            knn: Schilling + Henze; rev: reversed VRT; raw: MMD_raw, ED_raw; slow: MMDAgg b1500).
                                            T - F is the test time of the whole group for the seeds of that task.
  ssl_mmdagg5k_seeds.py (SimCLR MMDAgg b5000, CPU)  "ep N  mmdagg5000 r/n reject ... (feat F s, total T s)", n seeds.
  base_fill.py      (CIFAR and other models, stored features)  "--stat mmdagg|c2st ..." one statistic per task; per-epoch
                                            line ending in "(T s)" for the seeds of the task.
  other scripts     any line "(T s for n seeds)" or a per-epoch line ending in "(T s)" is taken with the --seeds of the
                                            command echoed in the log.
Per-seed seconds = (T - F) / seeds of the task. The table gives, per (script, group or stat, budget, device), the number
of epoch lines, the median and the 10th to 90th percentile of the per-seed seconds, the median per-epoch seconds for 50
seeds, and for SimCLR the median feature-extraction time. Files without a recognised timing line are counted at the end.
Usage (cluster, code folder):  python timing_summary.py [--logs '/scratch/p318770/ssl_supcon_code*/s50_logs/*.out' ...]
                               python timing_summary.py --list base     (prints the files behind one row, for inspection)
"""
import argparse, collections, glob, os, re, statistics as st

DEFAULT_LOGS = ["/scratch/p318770/ssl_supcon_code*/s50_logs/*.out", "/scratch/p318770/ssl_supcon_code*/*.out"]
RE_CMD = re.compile(r"python (\S+\.py)([^&;|\n]*)")
RE_FEAT = re.compile(r"^\s*ep\s+(\d+)\s+(.*?)\s*\(feat (\d+(?:\.\d+)?) s, total (\d+(?:\.\d+)?) s\)\s*$")
RE_FOR = re.compile(r"\((\d+(?:\.\d+)?) s for (\d+) seeds\)")
RE_EP = re.compile(r"^\s*ep\s*(\d+)\b.*\((\d+(?:\.\d+)?) s\)\s*$")
RE_SSL_HEAD = re.compile(r"^(\S+) (fast|vrt|base|slow|raw|rev|knn|extra): \d+ checkpoints to do .*seeds (\d+)-(\d+), device (\w+)")
RE_AGG_HEAD = re.compile(r"^(\S+): epochs \[.*\], seeds (\d+)-(\d+); (\d+) \(epoch, seed\) pairs to do, device (\w+)")
RE_DEV = re.compile(r"device (cuda|cpu)")


def n_seeds(spec):
    try:
        a, b = spec.split("-"); return int(b) - int(a) + 1
    except ValueError:
        return len(spec.split(","))


def arg(args, name, default=""):
    m = re.search(rf"--{name}\s+(\S+)", args)
    return m.group(1) if m else default


def parse_file(path):
    """list of (key, per_seed_seconds, per_epoch_seconds_50, feat_seconds or None)"""
    out = []
    cmd = None            # (script, label, budget, seeds, device)
    device = "?"
    for line in open(path, errors="replace"):
        m = RE_CMD.search(line)
        if m:
            script, a = m.group(1), m.group(2)
            label = arg(a, "stat") or arg(a, "group")
            budget = arg(a, "budgets") or arg(a, "budget") or arg(a, "n_samples") or arg(a, "n_sub")
            seeds = n_seeds(arg(a, "seeds", "42-91"))
            bn = " bn_train" if "--bn_train" in a else ""
            cmd = (script, label + bn, budget, seeds, device)
        d = RE_DEV.search(line)
        if d:
            device = d.group(1)
            if cmd:
                cmd = cmd[:4] + (device,)
        h = RE_SSL_HEAD.match(line)
        if h:
            cmd = ("ssl_s50_fill.py", h.group(2) + (" bn_train" if h.group(1).endswith("bntrain") else ""), "",
                   int(h.group(4)) - int(h.group(3)) + 1, h.group(5))
            continue
        h = RE_AGG_HEAD.match(line)
        if h:
            cmd = ("ssl_mmdagg5k_seeds.py", "mmdagg" + (" bn_train" if "bntrain" in path or "bn_train" in line else ""),
                   "5000", int(h.group(3)) - int(h.group(2)) + 1, h.group(5))
            continue
        if cmd is None:
            continue
        script, label, budget, seeds, dev = cmd
        m = RE_FEAT.match(line)
        if m:
            feat, total = float(m.group(3)), float(m.group(4))
            per_seed = (total - feat) / seeds
            if script == "ssl_mmdagg5k_seeds.py" or "mmdagg5000" in m.group(2):
                label2, budget2 = label, "5000"
            else:
                label2, budget2 = label, budget
            out.append(((script, label2, budget2, dev), per_seed, 50 * per_seed, feat))
            continue
        m = RE_FOR.search(line)
        if m:
            per_seed = float(m.group(1)) / int(m.group(2))
            out.append(((script, label, budget, dev), per_seed, 50 * per_seed, None))
            continue
        m = RE_EP.match(line)
        if m and "feat" not in line:
            per_seed = float(m.group(2)) / seeds
            out.append(((script, label, budget, dev), per_seed, 50 * per_seed, None))
    return out


def pct(xs, q):
    xs = sorted(xs); i = (len(xs) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def hms(s):
    return f"{s / 3600:.1f} h" if s >= 3600 else f"{s / 60:.1f} min" if s >= 60 else f"{s:.0f} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="*", default=DEFAULT_LOGS)
    ap.add_argument("--list", default="", help="print the files behind rows whose label or script contains this text")
    a = ap.parse_args()
    files = sorted({f for pat in a.logs for f in glob.glob(pat)})
    rows = collections.defaultdict(list); feats = collections.defaultdict(list); src = collections.defaultdict(set)
    unmatched = 0
    for f in files:
        got = parse_file(f)
        if not got:
            unmatched += 1; continue
        for key, ps, pe, ft in got:
            rows[key].append(ps); src[key].add(f)
            if ft is not None:
                feats[key].append(ft)
    print(f"{len(files)} log files, {len(files) - unmatched} with timing lines, {unmatched} without\n")
    print(f"{'script':22s} {'group / stat':18s} {'budget':>10s} {'dev':4s} {'epochs':>6s} {'s per seed (med, p10-p90)':>28s} "
          f"{'per epoch, 50 seeds':>20s} {'features':>10s}")
    for key in sorted(rows):
        v = rows[key]
        script, label, budget, dev = key
        feat = f"{hms(st.median(feats[key]))}" if feats[key] else ""
        print(f"{script:22s} {label:18s} {budget:>10s} {dev:4s} {len(v):6d} {st.median(v):11.1f} ({pct(v, .1):7.1f} -{pct(v, .9):8.1f}) "
              f"{hms(50 * st.median(v)):>20s} {feat:>10s}")
        if a.list and (a.list in label or a.list in script):
            for f in sorted(src[key]):
                print(f"      {f}")
    print("\nssl_s50_fill.py rows: the time covers every statistic of the group together (vrt: VRT b10000 and b1500; "
          "base: MMD, ED and C2ST at both budgets).\nPer-statistic numbers for the SimCLR tests need a separate timing "
          "run; base_fill.py and the standalone scripts are per statistic.")


if __name__ == "__main__":
    main()
