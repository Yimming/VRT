"""compare_ed_logs.py — compare the fast ED run (seed 42 rows of ed_<tag>_n1500.csv) with the paper-era
benchmark_v2 log of the same model, epoch by epoch.
Usage: python compare_ed_logs.py <old_log> <fast_csv> [seed=42]
Old log lines look like:  "  ep   0 gap=+0.0008  ED p=0.9621"
"""
import sys, re, csv
import numpy as np

log, fcsv = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

old = {}
for line in open(log):
    m = re.search(r"ep\s+(\d+).*?ED p=([0-9.]+)", line)
    if m:
        old[int(m.group(1))] = float(m.group(2))

new = {}
with open(fcsv) as fh:
    for r in csv.DictReader(fh):
        if int(r["seed"]) == seed:
            new[int(r["epoch"])] = float(r["p"])

common = sorted(set(old) & set(new))
if not common:
    sys.exit(f"no common epochs (old {len(old)}, new {len(new)})")
diff = np.array([new[e] - old[e] for e in common])
exact = int(np.sum(np.abs(diff) < 5e-5))          # old log has 4 decimals
print(f"{len(common)} common epochs (old log {len(old)}, fast csv seed {seed} {len(new)})")
print(f"identical to 4 decimals: {exact}/{len(common)}   max |diff| = {np.abs(diff).max():.4f}")
mism = [e for e in common if abs(new[e] - old[e]) >= 5e-5]
if mism:
    print("mismatches (epoch: old -> new):", ", ".join(f"{e}: {old[e]:.4f} -> {new[e]:.4f}" for e in mism[:40]))
    flips = [e for e in mism if (old[e] < 0.05) != (new[e] < 0.05)]
    print(f"decisions at alpha=0.05 that flip: {len(flips)}" + (f" at epochs {flips}" if flips else ""))
else:
    print("all decisions identical")
