"""
clean_chatstats_dups.py — remove duplicated, superseded and torn rows from the chat_stats result files (camera-ready
#27102, October 15 and 16).

The c-hat diagnostic (chat_stats_s50.py, 10 CPU tasks) was submitted twice (jobs 32818592 and 32822329) and both jobs ran
every task, appending to the same files. Consequences, read from the files on October 16:
  - every model's file holds most epochs twice (the second job skipped only the epochs the first had finished when it
    started); the repeated rows have identical values, the computation being deterministic;
  - at the flush boundaries of two concurrent appenders a line can be torn, so a few keys are repeated with a different
    value (zoo_c10_m4 epoch 17 seed 88 tr_q75: 61.391580 and 61.310344, the second a torn merge) or lost;
  - zoo_c10_m3 epochs 24, 27, 28 were left incomplete and were recomputed into chatstats_zoo_c10_m3_c24-28fix.csv
    (job 32832502); that file was cleaned on October 15 with --drop_epochs 24,27,28.
What the script does, per file (report only, or rewrite with --apply after copying the original to chat_stats/dup_backup/):
  1. rows of --drop_epochs are removed (only with --file; the fix file must hold all 50 seeds x 23 rows for each of them);
  2. lines that do not parse as tag,epoch,seed,stat,obs,p are removed;
  3. every remaining (epoch, seed, stat) key is kept once. Repeated keys with identical values keep the first row.
     Repeated keys with different values are resolved only for the summary rows tr_<q> / te_<q>: the value for which
     te_<q> - tr_<q> equals the obs of the matching statistic row (within 1e-5) is kept; a conflict on a statistic row
     (obs or p) or an unresolvable summary conflict blocks --apply for that file (recompute the epoch instead).
It also lists which epochs have seed-42 c-hat arrays in the _chat_s42.npz files (for the histograms).
Standard library only (numpy only for the npz listing, skipped when absent).
Usage (cluster, code folder, venv active):
  python clean_chatstats_dups.py --all                      # report for every chatstats_*_c*.csv in $S50/chat_stats
  python clean_chatstats_dups.py --all --apply              # rewrite them (files already clean are left untouched)
  python clean_chatstats_dups.py --file chatstats_zoo_c10_m3_c20-35.csv --drop_epochs 24,27,28 [--apply]
  python chat_stats_summary.py                              # afterwards: "rows: 0 duplicate ... 0 unparsable (clean)" everywhere
"""
import argparse, collections, glob, os, shutil, sys

B = os.environ.get("S50", "/home3/p318770/venvs/overfitting/benchmark_final/s50")
STATS = ["med", "mean", "tmean", "q10", "q25", "q75", "q90", "sd", "ks"]
SUMS = ["median", "mean", "sd", "q10", "q25", "q75", "q90"]
SUM_TO_STAT = {"median": "med", "mean": "mean", "sd": "sd", "q10": "q10", "q25": "q25", "q75": "q75", "q90": "q90"}
ROWS_PER_SEED = len(STATS) + 2 * len(SUMS)           # 23
SEEDS = set(range(42, 92))
HEADER = "tag,epoch,seed,stat,obs,p"
ALL_NAMES = STATS + [f"{a}_{b}" for a in ("tr", "te") for b in SUMS]


def parse_line(line, tag):
    """(epoch, seed, stat, obs, p) of a well-formed row of this tag, else None"""
    f = line.rstrip("\n").split(",")
    if len(f) != 6 or f[0] != tag:
        return None
    try:
        e, s = int(f[1]), int(f[2])
        o = float(f[4])
        p = float(f[5]) if f[5] != "" else None
    except ValueError:
        return None
    name = f[3]
    if name not in ALL_NAMES or (name in STATS) != (p is not None):
        return None
    return e, s, name, o, p


def scan(path, tag):
    """lines, rows per epoch -> {(seed, stat): [(line index, obs, p)]}, unparsable (line number, text)"""
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    bad = []
    with open(path) as fh:
        lines = fh.readlines()
    if not lines or lines[0].rstrip("\n") != HEADER:
        sys.exit(f"{path}: first line is not the header {HEADER}")
    for i, line in enumerate(lines[1:], start=1):
        r = parse_line(line, tag)
        if r is None:
            bad.append((i + 1, line.rstrip("\n")))
        else:
            e, s, name, o, p = r
            rows[e][(s, name)].append((i, o, p))
    return lines, rows, bad


def epoch_report(rows, bad, label):
    print(f"== {label}")
    for e in sorted(rows):
        keys = rows[e]
        n = sum(len(v) for v in keys.values())
        dup = sum(len(v) - 1 for v in keys.values())
        conf = sum(1 for v in keys.values() if len({(o, p) for _, o, p in v}) > 1)
        seeds = {s for s, _ in keys}
        complete = seeds == SEEDS and all((s, name) in keys for s in SEEDS for name in ALL_NAMES)
        if dup or conf or not complete:
            print(f"   epoch {e:3d}: {n:5d} rows, {len(keys):4d} keys, {dup:4d} repeated rows, {conf:2d} with a different value, "
                  f"{len(seeds)} seeds{' (complete)' if complete else ' (INCOMPLETE)'}")
    quiet = [e for e in sorted(rows) if not any(len(v) > 1 for v in rows[e].values())]
    if quiet:
        print(f"   epochs with every key exactly once: {quiet[0]} to {quiet[-1]}, {len(quiet)} of {len(rows)}" if len(quiet) > 1
              else f"   epoch with every key exactly once: {quiet[0]}")
    print(f"   unparsable lines: {len(bad)}")
    for i, t in bad[:10]:
        print(f"      line {i}: {t[:110]}{'...' if len(t) > 110 else ''}")


def resolve(key, cands, rows_e):
    """pick the consistent value for a repeated summary key with different values; None when it cannot be decided"""
    s, name = key
    if name in STATS or "_" not in name:
        return None
    side, q = name.split("_", 1)
    other = ("te_" if side == "tr" else "tr_") + q
    stat = SUM_TO_STAT.get(q)
    ov = {o for _, o, _ in rows_e.get((s, other), [])}
    sv = {o for _, o, _ in rows_e.get((s, stat), [])} if stat else set()
    if len(ov) != 1 or len(sv) != 1:
        return None
    o_other, d = ov.pop(), sv.pop()
    ok = [c for c in cands if abs((o_other - c[1] if side == "tr" else c[1] - o_other) - d) < 1e-5]
    return ok[0] if len(ok) == 1 else None


def npz_epochs(path):
    try:
        import numpy as np
    except ImportError:
        return None
    if not os.path.exists(path):
        return set()
    d = np.load(path)
    return {int(k[2:5]) for k in d.files if k.startswith("ep") and k.endswith("_tr") and k[:-3] + "_te" in d.files}


def clean_one(d, fname, drop, fix, apply):
    path = os.path.join(d, fname)
    tag = fname[len("chatstats_"):].rsplit("_c", 1)[0]
    lines, rows, bad = scan(path, tag)
    epoch_report(rows, bad, f"{fname} ({len(lines) - 1} data rows, tag {tag})")

    ok_fix = True
    if drop:
        fix_path = os.path.join(d, fix)
        if not os.path.exists(fix_path):
            sys.exit(f"fix file {fix_path} not found")
        _, frows, fbad = scan(fix_path, tag)
        epoch_report(frows, fbad, f"{fix} (must be complete for every dropped epoch)")
        ok_fix = not fbad and set(frows) >= set(drop) and all(
            len(frows[e]) == 50 * ROWS_PER_SEED and all(len(v) == 1 for v in frows[e].values()) for e in drop)
        print(f"== fix file covers {drop} completely with one row per key: {'yes' if ok_fix else 'NO'}")

    keep_idx = set()
    n_rep = n_res = 0
    unresolved = []
    for e in rows:
        if e in drop:
            continue
        for key, cands in rows[e].items():
            if len(cands) == 1:
                keep_idx.add(cands[0][0]); continue
            n_rep += len(cands) - 1
            if len({(o, p) for _, o, p in cands}) == 1:
                keep_idx.add(cands[0][0]); continue
            c = resolve(key, cands, rows[e])
            if c is None:
                unresolved.append((e,) + key + (tuple((o, p) for _, o, p in cands),))
                keep_idx.add(cands[0][0])
            else:
                n_res += 1
                keep_idx.add(c[0])
                print(f"   resolved: epoch {e} seed {key[0]} {key[1]} values {[o for _, o, _ in cands]} -> keep {c[1]} "
                      f"(consistent with the {SUM_TO_STAT[key[1].split('_', 1)[1]]} statistic)")
    n_drop = sum(len(v) for e in drop for v in rows.get(e, {}).values())
    kept = [lines[0]] + [lines[i] for i in sorted(keep_idx)]
    print(f"== cleaned file keeps {len(kept) - 1} of {len(lines) - 1} data rows ({n_drop} rows of epochs {drop or 'none'}, "
          f"{n_rep} repeated rows and {len(bad)} unparsable lines removed; {n_res} value conflicts resolved, {len(unresolved)} unresolved)")
    for u in unresolved[:10]:
        print(f"   UNRESOLVED: epoch {u[0]} seed {u[1]} {u[2]} values {u[3]}")
    after = collections.defaultdict(set)
    for i in keep_idx:
        e, s, name, _, _ = parse_line(lines[i], tag)
        after[e].add((s, name))
    inc = [e for e in sorted(after) if len(after[e]) != 50 * ROWS_PER_SEED]
    print(f"   epochs kept: {min(after)} to {max(after)}, {len(after)} epochs" + (f", INCOMPLETE after cleaning: {inc}" if inc else ", every kept epoch complete"))
    eps = npz_epochs(os.path.splitext(path)[0] + "_chat_s42.npz")
    miss = sorted(set(after) - eps) if eps is not None else None
    print(f"== seed-42 c-hat arrays in {os.path.basename(os.path.splitext(path)[0])}_chat_s42.npz: "
          + ("numpy not available" if eps is None else f"{len(eps)} epochs" + (f", MISSING {miss}" if miss else ", every kept epoch covered")))
    changed = len(kept) != len(lines)
    if not changed:
        print("   nothing to remove, file left untouched\n"); return
    if not apply:
        print("   report only; add --apply to rewrite (a copy of the original goes to chat_stats/dup_backup/)\n"); return
    if not ok_fix:
        sys.exit("not applied: the fix file does not cover the dropped epochs completely")
    if unresolved:
        print("   NOT APPLIED: unresolved value conflicts (recompute those epochs into a fix file and use --drop_epochs)\n"); return
    os.makedirs(os.path.join(d, "dup_backup"), exist_ok=True)
    bak = os.path.join(d, "dup_backup", fname)
    if os.path.exists(bak):
        sys.exit(f"not applied: backup {bak} exists already")
    shutil.copy2(path, bak)
    with open(path, "w") as fh:
        fh.writelines(kept)
    _, rows2, bad2 = scan(path, tag)
    dup2 = sum(len(v) - 1 for e in rows2 for v in rows2[e].values())
    print(f"   applied: original copied to dup_backup/; now {sum(len(v) for e in rows2 for v in rows2[e].values())} rows, "
          f"{dup2} repeated rows, {len(bad2)} unparsable lines, epochs {min(rows2)} to {max(rows2)} ({len(rows2)})\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="", help="one file name inside $S50/chat_stats")
    ap.add_argument("--all", action="store_true", help="every chatstats_*_c*.csv in $S50/chat_stats")
    ap.add_argument("--drop_epochs", default="", help="with --file: epochs whose rows are removed, e.g. 24,27,28")
    ap.add_argument("--fix_file", default="", help="file holding the recomputed epochs (default chatstats_<tag>_c<lo>-<hi>fix.csv)")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    d = f"{B}/chat_stats"
    if a.all:
        files = sorted(os.path.basename(f) for f in glob.glob(f"{d}/chatstats_*_c*.csv"))
        drop, fix = [], ""
    elif a.file:
        files = [a.file]
        drop = sorted(int(x) for x in a.drop_epochs.split(",")) if a.drop_epochs else []
        tag = a.file[len("chatstats_"):].rsplit("_c", 1)[0]
        fix = a.fix_file or (f"chatstats_{tag}_c{drop[0]}-{drop[-1]}fix.csv" if drop else "")
    else:
        sys.exit("give --file <name> or --all")
    for fname in files:
        clean_one(d, fname, drop, fix, a.apply)


if __name__ == "__main__":
    main()
