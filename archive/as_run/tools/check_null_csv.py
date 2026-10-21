"""check_null_csv.py — strict check of the raw CSV files behind the three CIFAR null harvests, nothing is summarized:
  null_cifar_vrt   vrt_null_*.csv    per (arch, bn_mode): every repetition has its 4 rows (split x budget), no NaN p,
                                     no duplicate (rep, split, budget) with a different p, p in [0, 1]
  null_cifar_base  base_null_*.csv   per (arch, bn_mode, split, method, budget): same checks per repetition
  null_cifar_rule  vrt_rule_*.csv    per (arch, budget, net): seeds exactly 42 to 91, no NaN, no conflicting duplicates
Prints one line per group with the counts and lists every anomaly; exit code 1 if any anomaly is found.
Usage:  python check_null_csv.py [S50 folder]
"""
import csv, glob, math, os, sys, collections

S50 = sys.argv[1] if len(sys.argv) > 1 else "/home3/p318770/venvs/overfitting/benchmark_final/s50"
PAPER_SEEDS = set(range(42, 92))
problems = []


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def scan(folder, pattern, key_fn, unit_fn, label):
    """key_fn(row) -> group key, unit_fn(row) -> unit key inside the group (e.g. (rep, split, budget)); returns
    {group: {unit: p}} after checking NaN and conflicting duplicates"""
    groups = collections.defaultdict(dict)
    files = sorted(f for f in glob.glob(os.path.join(S50, folder, pattern)) if "DRYRUN" not in f)
    if not files:
        problems.append(f"{label}: no files in {folder}"); return groups
    for f in files:
        if os.path.getsize(f) == 0:
            problems.append(f"{label}: empty file {os.path.basename(f)}"); continue
        with open(f) as fh:
            for ln, r in enumerate(csv.DictReader(fh), start=2):
                try:
                    g, u = key_fn(r), unit_fn(r)
                except (KeyError, ValueError) as e:
                    problems.append(f"{label}: unreadable row {os.path.basename(f)}:{ln} ({e})"); continue
                p = fnum(r.get("p"))
                if p is None or not (0.0 <= p <= 1.0):
                    problems.append(f"{label}: bad p {r.get('p')!r} at {os.path.basename(f)}:{ln} {g} {u}"); continue
                if u in groups[g] and abs(groups[g][u] - p) > 1e-12:
                    problems.append(f"{label}: conflicting duplicate {g} {u}: {groups[g][u]} vs {p} ({os.path.basename(f)}:{ln})")
                groups[g].setdefault(u, p)
    return groups


# 1. VRT on random networks
G = scan("null_cifar_vrt", "vrt_null_*.csv", lambda r: (r["arch"], r["bn_mode"]),
         lambda r: (int(r["rep"]), r["split"], r["budget"]), "vrt_null")
print("null_cifar_vrt (arch, bn_mode): repetitions complete / seen, missing rows")
for g in sorted(G):
    reps = collections.Counter(u[0] for u in G[g])
    full = sum(1 for c in reps.values() if c == 4)
    bad = {rep: c for rep, c in reps.items() if c != 4}
    gaps = sorted(set(range(max(reps) + 1)) - set(reps)) if reps else []
    print(f"  {g}: {full}/{len(reps)} complete, max rep {max(reps) if reps else '-'}" + (f", incomplete {bad}" if bad else "") + (f", gaps {gaps[:10]}" if gaps else ""))
    for rep, c in bad.items():
        problems.append(f"vrt_null {g}: repetition {rep} has {c} of 4 rows")

# 2. baselines on random networks
G = scan("null_cifar_base", "base_null_*.csv", lambda r: (r["arch"], r["bn_mode"], r["split"], r["method"], r["budget"]),
         lambda r: int(r["rep"]), "base_null")
print("null_cifar_base (arch, bn_mode, split, method, budget): repetitions")
for g in sorted(G):
    reps = sorted(G[g]); gaps = sorted(set(range(reps[-1] + 1)) - set(reps)) if reps else []
    print(f"  {g}: {len(reps)} reps, max rep {reps[-1] if reps else '-'}" + (f", gaps {gaps[:10]}" if gaps else ""))

# 3. blank control of the paper's rule
G = scan("null_cifar_rule", "vrt_rule_*.csv", lambda r: (r["arch"], r["budget"]), lambda r: (int(r["net"]), int(r["seed"])), "vrt_rule")
print("null_cifar_rule (arch, budget): networks with exactly the 50 paper seeds / networks seen")
for g in sorted(G):
    nets = collections.defaultdict(set)
    for net, seed in G[g]:
        nets[net].add(seed)
    ok = sum(1 for s in nets.values() if s == PAPER_SEEDS)
    for net, s in sorted(nets.items()):
        if s != PAPER_SEEDS:
            problems.append(f"vrt_rule {g}: network {net} seeds {sorted(s - PAPER_SEEDS)} outside 42-91, missing {sorted(PAPER_SEEDS - s)}")
    print(f"  {g}: {ok}/{len(nets)} networks with seeds 42-91 exactly")

print()
if problems:
    print(f"{len(problems)} anomalies:")
    for p in problems[:200]:
        print("  " + p)
    sys.exit(1)
print("no anomalies: every p is a number in [0, 1], no conflicting duplicates, rule seeds are exactly 42 to 91")
