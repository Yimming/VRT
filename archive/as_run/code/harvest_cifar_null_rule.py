"""harvest_cifar_null_rule.py — blank control of the paper's decision rule (cifar_null_rule.py): the paper's VRT
procedure (official split, analysis seeds 42 to 91, median p below 0.05) applied to untrained networks.
Per architecture and budget:
  flagged        networks the paper's rule would call a detection (median p over the seeds < 0.05)
  med p          median over the networks of their median p
  max seeds      the largest number of seeds with p < 0.05 for one network (the rule needs more than half)
  single tests   share of all single (network, seed) tests with p < 0.05
Usage:  python harvest_cifar_null_rule.py [out_dir] [-v]      (-v also lists every network)
"""
import sys, os, glob, csv, collections
import numpy as np

S50 = "/home3/p318770/venvs/overfitting/benchmark_final/s50"
args = [a for a in sys.argv[1:] if a != "-v"]
VERBOSE = "-v" in sys.argv[1:]
BF = args[0] if args else f"{S50}/null_cifar_rule"
ARCH_ORDER = ["r18_c100", "r50_c100", "simclr_c100", "r18_c10", "r50_c10", "simclr_c10", "frn_c10"]
NAME = {"r18_c10": "ResNet-18 CIFAR-10", "r18_c100": "ResNet-18 CIFAR-100", "r50_c100": "ResNet-50 CIFAR-100",
        "r50_c10": "ResNet-50 CIFAR-10", "simclr_c10": "SimCLR enc. CIFAR-10", "simclr_c100": "SimCLR enc. CIFAR-100",
        "frn_c10": "ResNet20-FRN CIFAR-10"}

P = collections.defaultdict(dict)        # (arch, budget, net) -> {seed: p}
for f in glob.glob(os.path.join(BF, "vrt_rule_*.csv")):
    if "DRYRUN" in f or os.path.getsize(f) == 0:
        continue
    for r in csv.DictReader(open(f)):
        P[(r["arch"], r["budget"], int(r["net"]))].setdefault(int(r["seed"]), float(r["p"]))
if not P:
    sys.exit(f"no result files in {BF}")

print(f"blank control: the paper's VRT decision on untrained networks (official split, the paper's seeds), {BF}\n")
print(f"{'model':22s} {'budget':>6s} {'networks':>8s} {'flagged':>8s} {'med p':>6s} {'max seeds':>9s} {'single tests':>12s}")
keys = sorted({(a, b) for a, b, _ in P}, key=lambda t: (ARCH_ORDER.index(t[0]) if t[0] in ARCH_ORDER else 99, t[1] != "large"))
for a, b in keys:
    nets = sorted(n for (a2, b2, n) in P if (a2, b2) == (a, b))
    meds, maxk, allp, seeds_n = [], 0, [], set()
    lines = []
    for n in nets:
        ps = np.array(list(P[(a, b, n)].values())); seeds_n.add(len(ps))
        meds.append(np.median(ps)); k = int((ps < 0.05).sum()); maxk = max(maxk, k); allp += list(ps)
        lines.append(f"    net {n:3d}  median p {np.median(ps):.3f}  {k}/{len(ps)} seeds < 0.05" + ("  FLAGGED" if np.median(ps) < 0.05 else ""))
    meds = np.array(meds); flagged = int((meds < 0.05).sum())
    ns = min(seeds_n)
    print(f"{NAME.get(a, a):22s} {'10,000' if b == 'large' else '1,500':>6s} {len(nets):8d} {f'{flagged}/{len(nets)}':>8s} "
          f"{np.median(meds):6.3f} {f'{maxk}/{ns}':>9s} {100 * np.mean(np.array(allp) < 0.05):11.1f}%"
          + ("" if len(seeds_n) == 1 else f"  (seeds per network {min(seeds_n)} to {max(seeds_n)})"))
    if VERBOSE:
        print("\n".join(lines))
print("\nflagged = median p over the seeds below 0.05, the paper's rule for a detection at a checkpoint;")
print("the 10,000 test images are the same for every seed and every network (the official test set)")
