"""export_fig_data.py — data files for the camera-ready figures from the 50-seed (s50) results.
Reads the s50 folders (vrt_fwd, reversed, mmd_p2000, ed_p2000, c2st, mmdagg) and writes small CSV files that the plotting
scripts read instead of the old 10-seed logs and single-seed benchmark files:
  vrt_<tag>.csv          epoch, vrt_p (median over the 50 seeds), n_seeds, n_reject (seeds with p < 0.05), dc (median)
                         10,000 tier (M = 2,000, k = 60); for every CIFAR model, the phase models and the two ImageNet models
  dc_<tag>.csv           epoch, dc_standard (train as reference), dc_reversed (test as reference), medians over the seeds
                         (Figure 8: v7, r50, imgnet_simple, imgnet_adv)
  base_<tag>.csv         epoch, mmd_p, ed_p, c2st_p, c2st_acc (medians), mmdagg_frac (share of seeds rejecting, every 5th
                         epoch) at the 10,000 tier (MMDAgg 5,000) (Figure 2: v7, r50; and ablA to ablD)
  fig4b_bn.csv           ImageNet MMD at 1,500 per split, share of the 50 seeds with p < 0.05 at epochs 200, 250, 300, eval-
                         and train-mode BatchNorm (the submitted Figure 4(b))
  fig4b_r50.csv          ResNet-50 at 10,000 per split, median p over the 50 seeds per epoch for MMD and ED on unnormalized
                         (raw) and l2-normalized (norm) features under eval-mode and train-mode BatchNorm extraction, and for
                         VRT under both modes (the camera-ready Figure 4(b)); n_seeds = seeds per epoch (min over the columns)
  onsets.csv             tag, sustained VRT onset under the paper's rule (median p < 0.05 at every later evaluated epoch)
Usage:  python export_fig_data.py <s50 folder> <output folder>
"""
import sys, os, glob, csv, collections
import numpy as np

S50, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
SEEDS = set(range(42, 92))
TAGS = ["v7", "r50", "ablA", "ablB", "ablC", "ablD", "wp", "phase_bs8", "phase_bs128", "phase_bs1024",
        "imgnet_simple", "imgnet_adv"] + [f"zoo_c10_m{i}" for i in range(5)] + [f"zoo_c100_m{i}" for i in range(5)]


def read(pattern, val, filt=lambda r: True, val2=None):
    d = collections.defaultdict(dict)                     # epoch -> seed -> value (or (value, value2))
    for f in glob.glob(pattern):
        for r in csv.DictReader(open(f)):
            if not filt(r) or int(r["seed"]) not in SEEDS:
                continue
            v = float(r[val]) if val2 is None else (float(r[val]), float(r[val2]))
            d[int(r["epoch"])].setdefault(int(r["seed"]), v)
    return d


def onset(eps, rej):
    return next((e for i, e in enumerate(eps) if all(rej[x] for x in eps[i:])), None)


ons = []
for tag in TAGS:
    pv = read(f"{S50}/vrt_fwd/vrt_fwd_{tag}_large_c*.csv", "p", filt=lambda r, t=tag: r["tag"] == t)
    dc = read(f"{S50}/vrt_fwd/vrt_fwd_{tag}_large_c*.csv", "dc", filt=lambda r, t=tag: r["tag"] == t)
    if not pv:
        print(f"{tag}: no VRT files"); continue
    eps = sorted(pv)
    with open(os.path.join(OUT, f"vrt_{tag}.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["epoch", "vrt_p", "n_seeds", "n_reject", "dc"])
        for e in eps:
            ps = np.array(list(pv[e].values()))
            w.writerow([e, f"{np.median(ps):.6f}", len(ps), int((ps < 0.05).sum()), f"{np.median(list(dc[e].values())):+.6f}"])
    rej = {e: np.median(list(pv[e].values())) < 0.05 for e in eps}
    o = onset(eps, rej)
    nseed = sorted({len(v) for v in pv.values()})
    ons.append([tag, o if o is not None else "", len(eps), f"{nseed[0]}-{nseed[-1]}"])
    print(f"{tag:14s} epochs {eps[0]}-{eps[-1]} ({len(eps)}), seeds {nseed[0]}-{nseed[-1]}, VRT onset {o if o is not None else '-'}")
with open(os.path.join(OUT, "onsets.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["tag", "vrt_onset", "n_epochs", "seeds_per_epoch"]); w.writerows(ons)

# Figure 8: dc under train- and test-as-reference
for tag in ("v7", "r50", "imgnet_simple", "imgnet_adv"):
    fw = read(f"{S50}/vrt_fwd/vrt_fwd_{tag}_large_c*.csv", "dc", filt=lambda r, t=tag: r["tag"] == t)
    rv = read(f"{S50}/reversed/vrt_rev_{tag}_large_c*.csv", "delta_c", filt=lambda r, t=tag: r["tag"] == t)
    with open(os.path.join(OUT, f"dc_{tag}.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["epoch", "dc_standard", "dc_reversed"])
        for e in sorted(set(fw) | set(rv)):
            a = f"{np.median(list(fw[e].values())):+.6f}" if e in fw else ""
            b = f"{np.median(list(rv[e].values())):+.6f}" if e in rv else ""
            w.writerow([e, a, b])

# Figure 2 (and Figure 3 if wanted): baselines per epoch, 10,000 tier
for tag in ("v7", "r50", "ablA", "ablB", "ablC", "ablD"):
    mmd = read(f"{S50}/mmd_p2000/mmd_{tag}_n10000_c*.csv", "p", filt=lambda r, t=tag: r["tag"] == t)
    ed = read(f"{S50}/ed_p2000/ed_{tag}_n10000_c*.csv", "p", filt=lambda r, t=tag: r["tag"] == t)
    c2 = read(f"{S50}/c2st/c2st_{tag}_b*.csv", "p", val2="acc", filt=lambda r, t=tag: r["model"] == t and r["budget"] == "b10000")
    ag = read(f"{S50}/mmdagg/mmdagg_{tag}_b5000*.csv", "p", filt=lambda r, t=tag: r["model"] == t and r["budget"] == "b5000")
    eps = sorted(set(mmd) | set(ed) | set(c2) | set(ag))
    with open(os.path.join(OUT, f"base_{tag}.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["epoch", "mmd_p", "ed_p", "c2st_p", "c2st_acc", "mmdagg_frac"])
        for e in eps:
            row = [e]
            for d in (mmd, ed):
                row.append(f"{np.median(list(d[e].values())):.6f}" if e in d else "")
            if e in c2:
                row += [f"{np.median([v[0] for v in c2[e].values()]):.6f}", f"{np.median([v[1] for v in c2[e].values()]):.6f}"]
            else:
                row += ["", ""]
            row.append(f"{np.mean(list(ag[e].values())):.4f}" if e in ag else "")
            w.writerow(row)

# Figure 4(b): ImageNet MMD at 1,500, eval vs train BatchNorm
with open(os.path.join(OUT, "fig4b_bn.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["model", "epoch", "eval_pct", "train_pct", "n_seeds"])
    for model in ("imgnet_adv", "imgnet_simple"):
        ev = read(f"{S50}/mmd_p2000/mmd_{model}_n1500_c*.csv", "p", filt=lambda r, t=model: r["tag"] == t)
        tr = read(f"{S50}/mmd_p2000/mmd_{model}_bntrain_n1500_c*.csv", "p", filt=lambda r, t=model + "_bntrain": r["tag"] == t)
        for e in (200, 250, 300):
            a, b = list(ev[e].values()), list(tr[e].values())
            w.writerow([model, e, f"{100 * np.mean(np.array(a) < 0.05):.0f}", f"{100 * np.mean(np.array(b) < 0.05):.0f}", f"{len(a)}/{len(b)}"])
# Figure 4(b), camera-ready: ResNet-50 under the three extraction settings (raw eval, normalized eval, normalized train)
cols = {"mmd_raw_eval": (f"{S50}/mmd_raw_p2000/mmd_r50_n10000_c*.csv", "r50"),
        "mmd_norm_eval": (f"{S50}/mmd_p2000/mmd_r50_n10000_c*.csv", "r50"),
        "mmd_norm_train": (f"{S50}/mmd_p2000/mmd_r50_bntrain_n10000_c*.csv", "r50_bntrain"),
        "mmd_raw_train": (f"{S50}/mmd_raw_p2000/mmd_r50_bntrain_n10000_c*.csv", "r50_bntrain"),
        "ed_raw_eval": (f"{S50}/ed_raw_p2000/ed_r50_n10000_c*.csv", "r50"),
        "ed_norm_eval": (f"{S50}/ed_p2000/ed_r50_n10000_c*.csv", "r50"),
        "ed_norm_train": (f"{S50}/ed_p2000/ed_r50_bntrain_n10000_c*.csv", "r50_bntrain"),
        "ed_raw_train": (f"{S50}/ed_raw_p2000/ed_r50_bntrain_n10000_c*.csv", "r50_bntrain"),
        "vrt_eval": (f"{S50}/vrt_fwd/vrt_fwd_r50_large_c*.csv", "r50"),
        "vrt_train": (f"{S50}/vrt_fwd/vrt_fwd_r50_bntrain_large_c*.csv", "r50_bntrain")}
data = {k: read(pat, "p", filt=lambda r, t=t: r["tag"] == t) for k, (pat, t) in cols.items()}
eps = sorted(set.intersection(*[set(d) for d in data.values()]))
with open(os.path.join(OUT, "fig4b_r50.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["epoch"] + list(cols) + ["n_seeds"])
    for e in eps:
        w.writerow([e] + [f"{np.median(list(data[k][e].values())):.6f}" for k in cols] + [min(len(data[k][e]) for k in cols)])
for k in cols:
    rej = {e: np.median(list(data[k][e].values())) < 0.05 for e in eps}
    o = onset(eps, rej)
    print(f"fig4b_r50 {k:15s} onset {o if o is not None else '-'}  (epochs {eps[0]}-{eps[-1]}, {len(eps)}; "
          f"seeds min {min(len(data[k][e]) for e in eps)})")
print(f"written to {OUT}")
