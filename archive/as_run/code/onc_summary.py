"""
onc_summary.py — MMD on the c-hat scores against VRT and against MMD on the embeddings, for every model that has
mmd_on_chat output (camera-ready #27102, October 13). Sustained onset over the evaluated epochs with all 50 seeds (median
p < 0.05 at that epoch and every later one). Also checks, seed by seed, that the dc written with the MMD-on-c-hat run equals
the dc of the s50 large-budget VRT (same c-hat values).
For the SimCLR models (tags simclr_c10, simclr_c100; ssl_mmd_on_chat.py) VRT and MMD on the embeddings come from the
Table 1 files ($S50/ssl/), and the check compares the VRT p written for the check seed with the Table 1 VRT p of the same
epoch and seed. --brackets (only once all 100 grid epochs of a model have 50 seeds) rewrites onc_simclr_brackets.txt with
the phase-2 task lines (every unevaluated epoch between the last non-rejecting evaluated epoch and the grid onset) and
prints the sbatch line; a second call after phase 2 should report "onset exact".
Every detecting model also gets an "after the onset" count (October 15): how many of the saved checkpoints after the
MMD-on-c-hat onset have been evaluated with 50 seeds. Saved checkpoints = the epochs of the VRT cell (vrt_fwd, or the
Table 1 file for SimCLR, 1 to 1,000). The sustained claim covers only evaluated checkpoints, so a cell is final when
this reads "every checkpoint evaluated" (SimCLR: phase 4, mmd_on_chat_simclr_phase4.txt, fills epochs 301 to 999).
Usage (cluster, code folder):  python onc_summary.py [--brackets]
"""
import collections, csv, glob, math, os, statistics as st, sys

B = os.environ.get("S50", "/home3/p318770/venvs/overfitting/benchmark_final/s50")
TAGS = (["r50", "v7", "ablA", "ablB", "ablC", "ablD"] + [f"zoo_c10_m{i}" for i in range(5)]
        + [f"zoo_c100_m{i}" for i in range(5)] + ["simclr_c10", "simclr_c100"]
        + ["wp", "imgnet_simple", "imgnet_adv", "phase_bs8", "phase_bs128", "phase_bs1024"])   # wp = the 5-layer CNN
SSL_DS = {"simclr_c10": "cifar10", "simclr_c100": "cifar100"}
CODE = "/scratch/p318770/ssl_supcon_code_s50"


def read(pattern, cols=("p",)):
    d = collections.defaultdict(dict)
    for f in glob.glob(pattern):
        for r in csv.DictReader(open(f)):
            try:
                d[int(r["epoch"])].setdefault(int(r["seed"]), tuple(float(r[c]) for c in cols))
            except (KeyError, ValueError):
                continue
    return d


def read_ssl(ds, stat):
    """Table 1 SimCLR file rows (eval mode, 10,000 points) of one statistic: epoch -> {seed: (p,)}"""
    d = collections.defaultdict(dict)
    for f in glob.glob(f"{B}/ssl/ssl_{ds}_*_c*.csv"):
        for r in csv.DictReader(open(f)):
            if r.get("model") == ds and r.get("stat") == stat and r.get("budget") == "b10000":
                d[int(r["epoch"])].setdefault(int(r["seed"]), (float(r["p"]),))
    return d


def onset(d):
    eps = sorted(e for e in d if len(d[e]) >= 50)
    rej = {e: st.median(v[0] for v in d[e].values()) < 0.05 for e in eps}
    return next((e for i, e in enumerate(eps) if all(rej[x] for x in eps[i:])), None), len(eps)


def fmt(x):
    return "-" if x is None else str(x)


print(f"{'model':13s} {'epochs':>11s} {'VRT':>5s} {'MMD on c-hat':>13s} {'MMD on emb.':>12s}  dc check")
groups = collections.defaultdict(list)
bracket_lines = []
for tag in TAGS:
    chat = read(f"{B}/mmd_on_chat/mmdchat_{tag}_c*.csv", ("p", "dc", "mmd_c"))
    if not chat:
        print(f"{tag:13s} no MMD-on-c-hat output yet"); continue
    nonfin = sum(1 for e in chat for s in chat[e] if not math.isfinite(chat[e][s][2]))
    if nonfin:                       # a NaN statistic gets p = 1/(n_perm + 1) from mmd_test_fast, so these p are invalid
        print(f"{tag:13s} INVALID: {nonfin} rows with a non-finite MMD statistic (their p is not a test result), no onset")
        continue
    if tag in SSL_DS:
        ds = SSL_DS[tag]
        vrt, emb = read_ssl(ds, "vrt"), read_ssl(ds, "mmd")
        chk = read(f"{B}/mmd_on_chat/mmdchat_{tag}_c*.csv", ("vrt_p",))
        pairs = [(e, s) for e in chk for s in chk[e] if e in vrt and s in vrt[e]]
        bad = sum(abs(chk[e][s][0] - vrt[e][s][0]) > 1e-6 for e, s in pairs)
        maxd = max((abs(chk[e][s][0] - vrt[e][s][0]) for e, s in pairs), default=0.0)
        flips = sum((chk[e][s][0] < 0.05) != (vrt[e][s][0] < 0.05) for e, s in pairs)
        what = f"VRT p of the check seed (max |dp| {maxd:.4f}, {flips} cross 0.05)"
    else:
        vrt = read(f"{B}/vrt_fwd/vrt_fwd_{tag}_large_c*.csv", ("p", "dc"))
        emb = read(f"{B}/mmd_p2000/mmd_{tag}_n10000*.csv")       # with or without a _c<lo>-<hi> suffix
        pairs = [(e, s) for e in chat for s in chat[e] if e in vrt and s in vrt[e]]
        bad = sum(abs(chat[e][s][1] - vrt[e][s][1]) > 1e-5 for e, s in pairs)
        what = "dc"
    (oc, nc), (ov, nv), (oe, ne) = onset(chat), onset(vrt), onset(emb)
    print(f"{tag:13s} {nc:>4d} / {nv:<4d} {fmt(ov):>5s} {fmt(oc):>13s} {fmt(oe):>12s}  {what}: {len(pairs)} pairs, {bad} mismatches")
    eps50 = {e for e in chat if len(chat[e]) >= 50}
    if oc is not None:                       # coverage of the saved checkpoints after the MMD-on-c-hat onset
        tail = [e for e in sorted(vrt) if len(vrt[e]) >= 50 and e > oc]
        miss = [e for e in tail if e not in eps50]
        print(f"{'':13s} after the onset {oc}: {len(tail) - len(miss)} of {len(tail)} saved checkpoints evaluated"
              + (f", {len(miss)} not yet ({miss[0]} to {miss[-1]})" if miss else ", every checkpoint evaluated, onset final"))
    ninf = read(f"{B}/mmd_on_chat/mmdchat_{tag}_c*.csv", ("n_inf",))          # files of the fixed script only
    rows = [(e, ninf[e][s][0], chat[e][s][0]) for e in ninf for s in ninf[e] if e in chat and s in chat[e]]
    if rows:
        def share(sel):
            return f"{sum(p < 0.05 for p in sel) / len(sel):.3f} of {len(sel)}" if sel else "none"
        capped = [p for e, k, p in rows if k > 0]
        plain = [p for e, k, p in rows if k == 0]
        early = lambda r: oc is None or r[0] < oc
        print(f"{'':13s} capped +inf c-hat in {len(capped)} of {len(rows)} rows (at most {int(max(k for _, k, _ in rows))} values "
              f"per row); p < 0.05 in rows with capping {share(capped)}, without {share(plain)}; before the onset "
              f"{share([p for r in rows if early(r) and r[1] > 0 for p in [r[2]]])} and "
              f"{share([p for r in rows if early(r) and r[1] == 0 for p in [r[2]]])}")
    if tag in SSL_DS and "--brackets" in sys.argv:
        eps = eps50
        grid_missing = [e for e in range(10, 1001, 10) if e not in eps]
        if grid_missing:
            print(f"{'':13s} phase 1 not complete ({100 - len(grid_missing)} of 100 grid epochs with 50 seeds), no bracket yet")
        elif oc is None:
            print(f"{'':13s} no sustained MMD-on-c-hat detection on the grid, no bracket needed")
        else:
            before = [e for e in sorted(eps) if e < oc]
            e0 = before[-1] if before else 0                     # last non-rejecting evaluated epoch
            gap = [e for e in range(e0 + 1, oc) if e not in eps]
            if gap:
                ds = SSL_DS[tag]
                bracket_lines.append(
                    f"cd {CODE} && python ssl_mmd_on_chat.py --ckpt_dir \"$(ls -d /scratch/p318770/model_zoos/ssl_supcon/SimCLR_{ds}_* | head -1)\" "
                    f"--dataset {ds} --epochs {gap[0]}-{gap[-1]} --seeds 42-91 --out {B}/mmd_on_chat/mmdchat_{tag}_c{gap[0]}-{gap[-1]}.csv")
                print(f"{'':13s} bracket: epochs {gap[0]} to {gap[-1]} ({len(gap)} not yet evaluated)")
            else:
                print(f"{'':13s} every epoch from {e0} to the onset {oc} is evaluated, onset exact")
    g = "zoo_c10" if tag.startswith("zoo_c10_") else "zoo_c100" if tag.startswith("zoo_c100_") else None
    if g:
        groups[g].append((ov, oc, oe))


def rng(xs):
    ys = [x for x in xs if x is not None]
    if not ys:
        return "-"
    txt = f"{min(ys)}-{max(ys)}" if min(ys) != max(ys) else str(ys[0])
    return txt + (f" ({len(ys)}/{len(xs)})" if len(ys) < len(xs) else "")


for g, rows in groups.items():
    print(f"{g + ' (x' + str(len(rows)) + ')':13s} {'':>11s} {rng([r[0] for r in rows]):>5s} {rng([r[1] for r in rows]):>13s} {rng([r[2] for r in rows]):>12s}")

if "--brackets" in sys.argv and bracket_lines:
    with open("onc_simclr_brackets.txt", "w") as f:              # rewritten on every call, never appended
        f.write("\n".join(bracket_lines) + "\n")
    print(f"\nonc_simclr_brackets.txt: {len(bracket_lines)} task line(s). Submit with\n"
          f"  sbatch --array=0-{len(bracket_lines) - 1} job_s50_gpu.sh onc_simclr_brackets.txt")
