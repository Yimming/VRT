"""
make_onsets.py — recompute every detection onset of the paper from the released per-seed results.

For each cell (model tag, extraction mode, statistic, budget) the script loads the per-seed results with the file routing
of code/find_grid_gaps.py, keeps the epochs at which all 50 analysis seeds (42 to 91) are present, applies the decision
rule of the paper and finds the sustained onset: the first evaluated epoch from which every later evaluated epoch
rejects. Decision rule per epoch: median p over the seeds below 0.05 (VRT, reversed VRT, MMD, ED, Schilling, Henze);
C2ST also needs median held-out accuracy above 0.55; MMDAgg rejects when at least 60 percent of the seeds reject (its
output is a binary decision per seed). The as-run reversed VRT and Henze runners wrote p with four decimals; the
exact permutation p-value is taken from the integer count columns when a file has them and otherwise recovered as
round(p x 2001)/2001 (2,000 permutations in every run of these families; pass --stored_p to use the four-decimal
values instead; the two differ in one onset, see README).

Output: onsets.csv with one row per cell (tag, mode, stat, budget, n_epochs, first_epoch, last_epoch, last_checkpoint,
onset, gap_before_onset = saved checkpoints between the last non-rejecting epoch and the onset that were not evaluated,
unevaluated_after_onset, borderline_epochs = evaluated epochs whose median p equals 0.05 exactly, partial_epochs =
epochs present with fewer than 50 seeds, status = final / provisional / null / "no complete epoch"; a cell with results
but no complete epoch is listed, never dropped), and on stdout the rows of Tables 1 to 3 in the layout of the paper (ranges
over the five zoo models, number of detecting models in brackets when fewer than five, "null" for no sustained
detection; the paper prints 100 for the Phase Zoo as "<= 100", its earliest checkpoint).

Before anything is aggregated, every result file of the families read here is validated against
tools/result_schemas.py (file names, row membership, admissible p-values and decisions, no key with two different
values across files); a violation stops the run, so that an invalid value can never become a silent "not rejected".
--no_validate skips that pass (for a tree that the export already validated).

Usage:
  python tools/make_onsets.py --data release_data                 # ssl_baselines is found inside the data folder
  python tools/make_onsets.py --data release_data --out onsets.csv --stored_p
"""
import argparse
import csv
import glob
import io
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from result_schemas import SchemaError, validate_tree  # noqa: E402

SEEDS = set(range(42, 92))
FAMILIES_READ = ["vrt_fwd", "reversed", "mmd_p2000", "ed_p2000", "mmd_raw_p2000", "ed_raw_p2000", "knn_v2", "c2st",
                 "bntrain_c2st", "mmdagg", "bntrain_mmdagg", "ssl", "ssl_bntrain", "ssl_baselines"]
TABLE_STATS = ["vrt", "mmd", "mmdagg", "c2st", "ed"]
GROUPS = [("ResNet20-FRN (CIFAR-10)", ["v7"]),
          ("ResNet-18 C100 Zoo (x5)", [f"zoo_c100_m{i}" for i in range(5)]),
          ("ResNet-18 C10 Zoo (x5)", [f"zoo_c10_m{i}" for i in range(5)]),
          ("ResNet-50 (CIFAR-100)", ["r50"]),
          ("Phase Zoo bs8", ["phase_bs8"]), ("Phase Zoo bs128", ["phase_bs128"]), ("Phase Zoo bs1024", ["phase_bs1024"]),
          ("SimCLR CIFAR-10", ["simclr_c10"]), ("SimCLR CIFAR-100", ["simclr_c100"]),
          ("A: baseline", ["ablA"]), ("B: 10x wd", ["ablB"]), ("C: strong aug", ["ablC"]), ("D: wd + aug", ["ablD"]),
          ("5-layer CNN (CIFAR-10)", ["wp"]),
          ("DeepResNet50 simple (ImageNet)", ["imgnet_simple"]), ("DeepResNet50 adv. (ImageNet)", ["imgnet_adv"])]
NO_BN = {"v7", "wp"}                                        # models without BatchNorm: eval mode only


def load_routes(data):
    os.environ["S50"] = data
    sslb = os.path.join(data, "ssl_baselines")
    os.environ["SSLB"] = sslb if os.path.isdir(sslb) else os.environ.get("SSLB", sslb)
    sys.path.insert(0, os.path.join(ROOT, "code"))
    import find_grid_gaps                                   # noqa: E402  (reads S50 / SSLB at import)
    from release_utils import exact_perm_p                  # noqa: E402
    return find_grid_gaps, exact_perm_p


def decide(stat, vals, where=""):
    """decision rule of the paper; refuses values that cannot be a p-value, decision or accuracy"""
    ps = [p for p, _ in vals]
    if stat == "mmdagg":
        if any(p not in (0.0, 1.0) for p in ps):
            raise ValueError(f"{where}: MMDAgg decisions must be 0 or 1")
        return sum(ps) / len(ps) >= 0.6
    lo = 0.0 if stat in ("c2st", "schilling") else 1e-12          # normal-approximation p may round to 0; permutation p > 0
    if any(not (lo <= p <= 1) for p in ps):
        raise ValueError(f"{where}: p-values outside (0, 1] (NaN would count as 'not rejected')")
    if stat == "c2st":
        accs = [a for _, a in vals]
        if any(not (0 <= a <= 1) for a in accs):
            raise ValueError(f"{where}: C2ST accuracies outside [0, 1]")
        return st.median(ps) < 0.05 and st.median(accs) > 0.55
    return st.median(ps) < 0.05


def exact_p_cell(routes, exact_perm_p, tag, mode, stat, N):
    """epoch -> {seed: p_exact} for the two four-decimal families of the CIFAR-style models. New-format rows carry the
    integer permutation count and n_perm, and p_exact = (count + 1)/(n_perm + 1) is taken from them; as-run rows hold
    p with four decimals and no count, and the exact value is round(p x 2001)/2001 (every historical run of these two
    families used 2,000 permutations, which the protocol validation enforces where the column exists)."""
    t = tag if mode == "eval" else f"{tag}_bntrain"
    if stat == "vrt_rev":
        pats, pcol, ccol = [f"{routes.BF}/reversed/vrt_rev_{t}_{'large' if N == 10000 else 'small'}_c*.csv"], "p", "count"
    else:
        pats, pcol, ccol = [f"{routes.BF}/knn_v2/knn_{t}_n{N}_part*.csv"], "p_henze", "henze_count"
    out = {}
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            raw = open(f, "rb").read().replace(b"\x00", b"")
            for r in csv.DictReader(io.StringIO(raw.decode())):
                ep, sd = int(r["epoch"]), int(r["seed"])
                if r.get(ccol) not in (None, "") and r.get("n_perm") not in (None, ""):
                    px = (int(r[ccol]) + 1) / (int(r["n_perm"]) + 1)
                else:
                    px = exact_perm_p(float(r[pcol]), 2000)
                out.setdefault(ep, {}).setdefault(sd, px)
    return out


def analyse(routes, exact_perm_p, tag, mode, stat, N, cps, stored_p):
    d = routes.cell(tag, mode, stat, N)
    if not d:
        return None
    comp = sorted(e for e in d if set(d[e]) >= SEEDS)
    partial = [e for e in sorted(d) if e not in comp]
    row = {"tag": tag, "mode": mode, "stat": stat, "budget": N, "n_epochs": len(comp), "first_epoch": "",
           "last_epoch": "", "last_checkpoint": cps[-1] if cps else "", "onset": "", "gap_before_onset": "",
           "unevaluated_after_onset": "", "borderline_epochs": "", "partial_epochs": len(partial), "status": ""}
    if not comp:                                            # results exist but no epoch has all 50 seeds: say so
        row["status"] = f"no complete epoch ({len(partial)} epochs with fewer than 50 seeds)"
        return row
    vals = {e: [d[e][s] for s in sorted(SEEDS)] for e in comp}
    if stat in ("vrt_rev", "henze") and not stored_p:
        if tag in routes.SSL:                                # SimCLR files hold six-decimal p (exact on the 1/2001 lattice)
            vals = {e: [(exact_perm_p(p, 2000), a) for p, a in v] for e, v in vals.items()}
        else:
            px = exact_p_cell(routes, exact_perm_p, tag, mode, stat, N)
            vals = {e: [(px[e][s], a) for s, (p, a) in zip(sorted(SEEDS), v)] for e, v in vals.items()}
    rej = {e: decide(stat, vals[e], f"{tag} {mode} {stat} {N} epoch {e}") for e in comp}
    e1 = next((e for i, e in enumerate(comp) if all(rej[x] for x in comp[i:])), None)
    borderline = [e for e in comp if stat != "mmdagg" and st.median(p for p, _ in vals[e]) == 0.05]
    row.update(first_epoch=comp[0], last_epoch=comp[-1], onset="" if e1 is None else e1,
               borderline_epochs=";".join(map(str, borderline)))
    if e1 is not None and cps:
        earlier = [e for e in comp if e < e1]
        e0 = earlier[-1] if earlier else None
        row["gap_before_onset"] = sum(1 for c in cps if (e0 is None or c > e0) and c < e1 and c not in comp)
        row["unevaluated_after_onset"] = sum(1 for c in cps if c > e1 and c not in comp)
        row["status"] = "final" if row["gap_before_onset"] == 0 and row["unevaluated_after_onset"] == 0 else "provisional"
    elif cps:
        row["unevaluated_after_onset"] = sum(1 for c in cps if c > comp[-1])
        row["status"] = "null, last checkpoint evaluated" if row["unevaluated_after_onset"] == 0 else "null, provisional"
    return row


def fmt(onsets):
    """paper format for a group of models: null, N, min--max, with (k) when fewer than all detect"""
    present = [o for o in onsets if o is not None]
    if not present:
        return "null"
    s = str(min(present)) if min(present) == max(present) else f"{min(present)}--{max(present)}"
    return s + (f" ({len(present)})" if len(present) < len(onsets) else "")


def print_tables(rows):
    look = {(r["tag"], r["mode"], r["stat"], r["budget"]): (None if r["onset"] == "" else r["onset"]) for r in rows}

    def val(tags, mode, stat):
        N = 5000 if stat == "mmdagg" else 10000
        got = [look.get((t, mode, stat, N), "missing") for t in tags]
        if "missing" in got:
            return "missing"
        return fmt(got)

    print("\nTable 1 (eval mode): " + " | ".join(TABLE_STATS))
    for label, tags in GROUPS:
        print(f"  {label:32s} " + " | ".join(val(tags, "eval", s) for s in TABLE_STATS))
    print("\nTable 2 (eval -> train, BatchNorm models): " + " | ".join(TABLE_STATS))
    for label, tags in GROUPS:
        if tags[0] in NO_BN:
            continue
        print(f"  {label:32s} " + " | ".join(f"{val(tags, 'eval', s)} -> {val(tags, 'train', s)}" for s in TABLE_STATS))
    print("\nTable 3 (unnormalized features, eval -> train): mmd_raw | ed_raw")
    for label, tags in GROUPS:
        cells = []
        for s in ("mmd_raw", "ed_raw"):
            cells.append(val(tags, "eval", s) if tags[0] in NO_BN else f"{val(tags, 'eval', s)} -> {val(tags, 'train', s)}")
        print(f"  {label:32s} " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with the result families (vrt_fwd, mmd_p2000, ssl, ...)")
    ap.add_argument("--out", default="onsets.csv")
    ap.add_argument("--stored_p", action="store_true", help="use the four-decimal p of reversed VRT and Henze as stored")
    ap.add_argument("--no_validate", action="store_true", help="skip the schema validation of the result files")
    ap.add_argument("--no_protocol_check", action="store_true",
                    help="do not require the fixed protocol values (direction, M, k, n_perm) in the result rows")
    a = ap.parse_args()
    data = os.path.abspath(os.path.expanduser(a.data))
    if not a.no_validate:
        try:
            summary = validate_tree(data, families=FAMILIES_READ, protocol=not a.no_protocol_check)
        except SchemaError as e:
            raise SystemExit(f"result files not valid: {e}")
        print(f"validated {summary['files']} files, {summary['records']} records, {summary['legacy']} legacy files skipped, "
              f"{summary['cross_file_duplicates']} keys repeated across files with equal values")
    routes, exact_perm_p = load_routes(data)
    rows = []
    for tag in routes.CIFAR + routes.OTHER + list(routes.SSL):
        cps = routes.checkpoints(tag)
        if not cps:
            print(f"{tag}: no VRT results found, skipped", file=sys.stderr)
            continue
        for mode in ("eval", "train"):
            for stat, N in routes.STATS_LARGE + routes.STATS_SMALL:
                row = analyse(routes, exact_perm_p, tag, mode, stat, N, cps, a.stored_p)   # errors propagate
                if row:
                    rows.append(row)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    n_gap = sum(1 for r in rows if r["gap_before_onset"] not in ("", 0))
    n_border = sum(1 for r in rows if r["borderline_epochs"])
    n_partial = sum(1 for r in rows if r["partial_epochs"])
    incomplete = [r for r in rows if r["n_epochs"] == 0]
    print(f"{len(rows)} cells written to {a.out} ({len(rows) - len(incomplete)} with at least one complete epoch, "
          f"{len(incomplete)} with results but no complete epoch); {n_gap} with unevaluated checkpoints before the onset, "
          f"{n_border} with an epoch at median p = 0.05 exactly, {n_partial} with epochs missing seeds"
          + (" (four-decimal p as stored)" if a.stored_p else " (exact p for reversed VRT and Henze)"))
    for r in incomplete:
        print(f"  incomplete: {r['tag']} {r['mode']} {r['stat']} {r['budget']}: {r['status']}")
    for r in rows:
        if r["borderline_epochs"]:
            print(f"  borderline: {r['tag']} {r['mode']} {r['stat']} {r['budget']} epochs {r['borderline_epochs']}")
    print_tables(rows)


if __name__ == "__main__":
    main()
