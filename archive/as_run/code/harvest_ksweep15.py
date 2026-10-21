"""harvest_ksweep15.py — sustained onset of the small-budget VRT (k=15) from ksweep15_*.csv files.
Usage: python harvest_ksweep15.py <dir>"""
import sys, glob, os, csv, collections
import numpy as np
root = sys.argv[1]
rows = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(os.path.join(root, "ksweep15_*.csv")):
    for r in csv.DictReader(open(f)):
        rows[r["model"]][int(r["epoch"])].append(float(r["p"]))
print(f"{'unit':14s} {'epochs':>6s} {'first':>6s} {'onset':>6s}")
for unit in sorted(rows):
    eps = sorted(rows[unit]); med = {e: float(np.median(rows[unit][e])) for e in eps}
    first = next((e for e in eps if med[e] < 0.05), None)
    onset = next((e for i, e in enumerate(eps) if all(med[x] < 0.05 for x in eps[i:])), None)
    f = lambda x: "--" if x is None else str(x)
    print(f"{unit:14s} {len(eps):6d} {f(first):>6s} {f(onset):>6s}")
