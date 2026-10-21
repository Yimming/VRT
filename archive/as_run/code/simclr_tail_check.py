"""
simclr_tail_check.py — SimCLR onsets recomputed from whatever tail epochs exist now, against the values in the paper
(camera-ready #27102, October 12).

For every SimCLR cell (mode, budget, statistic) it prints the sustained onset from the current result files, how many of the
checkpoints after that onset have been evaluated (final when all of them), and the paper value; for every cell whose onset
moved it lists the evaluated epochs between the paper onset and the new onset that do not reject (median p, or the MMDAgg
fraction of rejecting seeds), which is why the onset moved.
Uses the loaders and decision rules of find_grid_gaps.py (same folder): 50 seeds per evaluated epoch, median p < 0.05,
C2ST also median accuracy > 0.55, MMDAgg at least 60 percent of the seeds rejecting.
Usage on the cluster (code folder):  python simclr_tail_check.py
On a laptop with a copy of the result folders (ssl/, ssl_bntrain/ and ssl_baselines/ inside <data>):
  S50=<data> SSLB=<data>/ssl_baselines python3 simclr_tail_check.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import find_grid_gaps as fg

# reference values of October 12 (commit 0c5e3b1 for Tables 2 and 3, the kNN table and Appendix G; Table 1 for eval; the
# 1,500-point tier of October 10); "-" = no sustained detection. Release note: this monitoring script was kept as run; its
# SPEC is NOT the camera-ready (e.g. the train-mode MMDAgg 5,000 onsets are 632/495 in Table 2, so the script prints
# "CHANGED" for them), and the tails evaluated later moved several provisional small-budget values. The reference for the
# published numbers is tools/make_onsets.py on data/ (docs/onsets_final.csv).
SPEC = """eval 10000 vrt 229 191; eval 10000 mmd 284 213; eval 5000 mmdagg 483 348; eval 10000 c2st 691 653; eval 10000 ed 283 213;
eval 10000 mmd_raw 100 127; eval 10000 ed_raw 100 127; eval 10000 henze 485 436; eval 10000 schilling 485 436; eval 10000 vrt_rev 256 208;
train 10000 vrt 527 324; train 10000 mmd 690 579; train 5000 mmdagg 616 490; train 10000 c2st 758 744; train 10000 ed 687 579;
train 10000 mmd_raw 646 405; train 10000 ed_raw 540 403; train 10000 henze 644 641; train 10000 schilling 644 641;
eval 1500 vrt 689 465; eval 1500 mmd 675 504; eval 1500 mmdagg 713 581; eval 1500 c2st 810 741; eval 1500 ed 679 504;
eval 1500 henze 734 684; eval 1500 schilling 731 684; eval 1500 vrt_rev 679 495;
train 1500 vrt 846 674; train 1500 mmd 946 801; train 1500 ed 951 801; train 1500 c2st - -; train 1500 henze 954 897;
train 1500 schilling 943 897"""


def describe(stat, vals):
    ps = np.array([p for p, _ in vals])
    if stat == "mmdagg":
        return f"fraction {ps.mean():.2f}"
    txt = f"median p {np.median(ps):.3f}"
    if stat == "c2st":
        txt += f", median acc {np.median([a for _, a in vals]):.3f}"
    return txt


def main():
    n_changed = n_prov = 0
    print(f"{'cell':40s} {'paper':>6s} {'now':>6s}  evaluated after onset   status")
    for item in SPEC.replace("\n", " ").split(";"):
        mode, N, stat, a, b = item.split(); N = int(N)
        for tag, want in (("simclr_c10", a), ("simclr_c100", b)):
            want = None if want == "-" else int(want)
            d = fg.cell(tag, mode, stat, N)
            comp = sorted(e for e in d if len(d[e]) >= fg.NS)
            rej = {e: fg.rejects(stat, list(d[e].values())) for e in comp}
            e1 = next((e for i, e in enumerate(comp) if all(rej[x] for x in comp[i:])), None)
            after = sum(1 for e in comp if e1 and e > e1); total = 1000 - e1 if e1 else 0
            final = (after == total)
            status = ("final" if final else "provisional") + ("" if e1 == want else ", CHANGED")
            n_changed += e1 != want; n_prov += not final
            name = f"{tag} {mode} {stat} {N}"
            print(f"{name:40s} {str(want):>6s} {str(e1):>6s}  {after:>4d} / {total:<4d}            {status}")
            if e1 != want and want is not None and e1 is not None and e1 > want:
                bad = [e for e in comp if want <= e < e1 and not rej[e]]
                for e in bad:
                    print(f"{'':44s}epoch {e}: does not reject ({describe(stat, list(d[e].values()))})")
    print(f"\n{n_changed} cells changed, {n_prov} cells still provisional (tail not fully evaluated)")


if __name__ == "__main__":
    main()
