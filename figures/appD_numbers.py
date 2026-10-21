"""
appD_numbers.py — every number of the synthetic power panel (Figure 3(a) of the camera-ready, file fig4_diagnosis.pdf;
the panel shows the first, second and fourth group below) and of the Appendix D tables, from the three synthetic result
files (camera-ready #27102, October 9). Rejections at alpha = 0.05 out of the repetitions, with the percentage used in
the text. The group list below is the five-group version of the panel that preceded the final three-group layout; the
remaining groups are tabulated in Appendix D.

Inputs: fig_data_s50/fig4a_clusters.csv and fig4a_matched.csv, plus
the Appendix D mmd_bandwidth.csv located through results.family_path.
Usage: python figures/appD_numbers.py [--latex]
       python figures/appD_numbers.py --dir INPUT_DIR --bandwidth BANDWIDTH_CSV
"""
import argparse
import collections
import csv
import os
import sys
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from results import family_path


def load(path, key):
    """{key(row): [p, ...]}"""
    out = collections.defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            out[key(r)].append(float(r["p"]))
    return out


def cell(ps, latex=False):
    """count of rejections; the percentage is shown with one decimal (199/200 is 99.5%, not 100%)"""
    if not ps:
        return "-"
    n = sum(p < 0.05 for p in ps)
    return f"{n}/{len(ps)}" if latex else f"{n:4d}/{len(ps):<4d} ({100 * n / len(ps):5.1f}%)"


def pct(ps):
    """percentage of rejections with one decimal (the paper's text rounds to integers where it says so)"""
    return round(100 * sum(p < 0.05 for p in ps) / len(ps), 1) if ps else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "fig_data_s50"))
    ap.add_argument("--bandwidth", default=str(family_path(ROOT.parent / "data", "synth") / "mmd_bandwidth.csv"), help="Path to mmd_bandwidth.csv")
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    bw_path = a.bandwidth or os.path.join(a.dir, "mmd_bandwidth.csv")
    for f in (os.path.join(a.dir, "fig4a_clusters.csv"), os.path.join(a.dir, "fig4a_matched.csv"), bw_path):
        if not os.path.exists(f):
            raise SystemExit(f"missing input {f} (fig4a_clusters.csv and fig4a_matched.csv: synth_fig4a_clusters.py / "
                             f"synth_fig4a_matched.py; mmd_bandwidth.csv: mmd_bandwidth_control.py, pass --bandwidth)")
    CL = load(os.path.join(a.dir, "fig4a_clusters.csv"), lambda r: (int(r["C"]), r["cond"], r["test"]))
    MA = load(os.path.join(a.dir, "fig4a_matched.csv"), lambda r: (int(r["budget"]), r["cond"], r["test"]))
    BW = load(bw_path, lambda r: (r["cond"], float(r["scale"]), r["test"]))

    print("== Synthetic power panel (Figure 3(a) shows groups 1, 2 and 4), percent rejections (VRT, MMD)")
    fig = [("within-class concentration, training +5%", CL[(10, "tight_0.05", "VRT")], CL[(10, "tight_0.05", "MMD")]),
           ("concentrated subpopulation, 10% of training at 3 kappa_w", CL[(10, "memo_0.10", "VRT")], CL[(10, "memo_0.10", "MMD")]),
           ("single-cloud concentration, kappa 250 -> 256.7", MA[(5000, "conc_0.3", "VRT")], MA[(5000, "conc_0.3", "MMD")]),
           ("single-cloud mean direction, 2 degrees", MA[(5000, "mean_2deg", "VRT")], MA[(5000, "mean_2deg", "MMD")]),
           ("radial spread, log-sd 0.05 -> 0.20", MA[(5000, "S2_radial_tail", "VRT")], MA[(5000, "S2_radial_tail", "MMD")])]
    for name, v, m in fig:
        print(f"   {name:58s} VRT {cell(v)}   MMD {cell(m)}")

    print("\n== Class-cluster model (D = 500, N = 5,000 per sample, kappa_w = 2,500, matched budget, 2,000 permutations)")
    rows_cl = [(10, "null", "Null, both samples kappa_w"), (10, "tight_0.05", "Within-class concentration, training +5%"),
               (10, "memo_0.10", "Concentrated subpopulation, 10% at 3 kappa_w"), (10, "shift_4", "Per-class location shift, 4 degrees"),
               (100, "null", "Null, C = 100"), (100, "tight_0.05", "Within-class concentration +5%, C = 100"),
               (100, "memo_0.10", "Concentrated subpopulation 10%, C = 100"), (100, "shift_4", "Per-class shift 4 degrees, C = 100")]
    for C, c, name in rows_cl:
        print(f"   C={C:3d} {name:48s} VRT {cell(CL[(C, c, 'VRT')])}   MMD {cell(CL[(C, c, 'MMD')])}"
              f"   median p VRT {st.median(CL[(C, c, 'VRT')]) if CL[(C, c, 'VRT')] else float('nan'):.3f}"
              f" MMD {st.median(CL[(C, c, 'MMD')]) if CL[(C, c, 'MMD')] else float('nan'):.3f}")
    print("   tightening sweep (C = 10):")
    for rel in ("0.03", "0.05", "0.07", "0.10", "0.30"):
        c = "tight_" + rel
        print(f"      +{int(float(rel) * 100):2d}%   VRT {cell(CL[(10, c, 'VRT')])}   MMD {cell(CL[(10, c, 'MMD')])}")

    print("\n== Single cloud (fig4a_matched.csv), rejections out of 200 at each budget; VRT, MMD, norm tests")
    conds = [("null_vmf", "Null, vMF(250) against vMF(250)"), ("conc_0.3", "Concentration, kappa 250 -> 256.7"),
             ("mean_2deg", "Mean direction, 2 degrees"), ("mix_0.05", "Mixture, 5% of vMF(750)"),
             ("S0_null", "Radial null"), ("S1_radial_mean", "Radial scale x1.03"), ("S2_radial_tail", "Radial spread 0.05 -> 0.20"),
             ("S3_angular", "Concentration with radii"), ("S4_angular_plus_radial", "Concentration with radii and scale x1.03")]
    for b in (5000, 1500, 500):
        print(f"   budget {b}")
        for c, name in conds:
            parts = [f"{t} {cell(MA[(b, c, t)])}" for t in ("VRT", "MMD", "NORM-MED", "NORM-DISP") if MA.get((b, c, t))]
            print(f"      {name:44s} " + "   ".join(parts))

    print("\n== Fixed-bandwidth control (C = 10, mmd_bandwidth.csv)")
    sig = [float(r["sigma"]) for r in csv.DictReader(open(bw_path))]
    dw = [float(r["d_within"]) for r in csv.DictReader(open(bw_path))]
    db = [float(r["d_between"]) for r in csv.DictReader(open(bw_path))]
    print(f"   median-heuristic sigma {st.median(sig):.3f} (min {min(sig):.3f}, max {max(sig):.3f}); median within-class distance "
          f"{st.median(dw):.3f}; between-class {st.median(db):.3f}")
    scales = sorted({k[1] for k in BW if k[2] == "MMD"})
    for c in ("null", "tight_0.05"):
        parts = [f"VRT {cell(BW[(c, 0.0, 'VRT')])}"] + [f"MMD x{s:g} {cell(BW[(c, s, 'MMD')])}" for s in scales]
        print(f"   {c:12s} " + "   ".join(parts))

    if a.latex:
        print("\n== LaTeX rows (rejections out of 200; VRT & MMD)")
        print("% class-cluster table")
        for C, c, name in rows_cl:
            print(f"{name} & {cell(CL[(C, c, 'VRT')], True)} & {cell(CL[(C, c, 'MMD')], True)} \\\\")
        print("% tightening sweep")
        print("Training tightening (percent) & " + " & ".join(f"+{int(float(r) * 100)}" for r in ("0.03", "0.05", "0.07", "0.10", "0.30")) + " \\\\")
        print("VRT & " + " & ".join(cell(CL[(10, 'tight_' + r, 'VRT')], True) for r in ("0.03", "0.05", "0.07", "0.10", "0.30")) + " \\\\")
        print("MMD & " + " & ".join(cell(CL[(10, 'tight_' + r, 'MMD')], True) for r in ("0.03", "0.05", "0.07", "0.10", "0.30")) + " \\\\")
        print("% single cloud, budget 5000")
        for c, name in conds[:7]:
            parts = [cell(MA[(5000, c, t)], True) if MA.get((5000, c, t)) else "--" for t in ("VRT", "MMD")]
            print(f"{name} & " + " & ".join(parts) + " \\\\")
        print("% bandwidth control")
        for c in ("null", "tight_0.05"):
            print(f"{c} & {cell(BW[(c, 0.0, 'VRT')], True)} & " + " & ".join(cell(BW[(c, s, 'MMD')], True) for s in scales) + " \\\\")


if __name__ == "__main__":
    main()
