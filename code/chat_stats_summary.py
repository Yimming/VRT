"""
chat_stats_summary.py — reads $S50/chat_stats/chatstats_<tag>_c*.csv (chat_stats_s50.py) and prints, per model, the median
p over the 50 seeds of every statistic on the c-hat values at each evaluated epoch, the onset of each statistic inside the
evaluated window (first evaluated epoch from which every later evaluated epoch has median p < 0.05; a window onset, not
the sustained onset over the whole run), the whole-run onsets of VRT (vrt_fwd) and of MMD on c-hat (mmd_on_chat) for
comparison, and the median over seeds of the test-minus-train difference of each c-hat summary (which part of the
distribution moves). Protocol checks per model: every evaluated epoch has exactly the seeds 42 to 91, and for every
(epoch, seed) both the median difference and its p equal the dc and p of vrt_fwd_<tag>_large_c*.csv; pairs that are
missing on either side are reported, not skipped. Rows are also checked for duplicates (October 15): a (epoch, seed,
stat) key that occurs more than once across the files of a model is counted, and counted again as a conflict when the
two values differ (the C10 Zoo m3 diagnostic was submitted twice and both jobs appended to the same file); the last file
in sorted order wins, and unparsable rows are counted instead of raising. clean_chatstats_dups.py removes such rows.
Usage (cluster, code folder):  python chat_stats_summary.py [--tags v7,ablA]
"""
import argparse, collections, csv, glob, os, statistics as st

B = os.environ.get("S50", "/home3/p318770/venvs/overfitting/benchmark_final/s50")
STATS = ["med", "mean", "tmean", "q10", "q25", "q75", "q90", "sd", "ks"]
SUMS = ["median", "mean", "sd", "q10", "q25", "q75", "q90"]
SEEDS = set(range(42, 92))


def onset(med):
    eps = sorted(med); rej = {e: med[e] < 0.05 for e in eps}
    return next((e for i, e in enumerate(eps) if all(rej[x] for x in eps[i:])), None)


def read_seed_p(pattern, pcol="p", extra=None):
    """epoch -> seed -> p (and extra column) from one family of result files"""
    d = collections.defaultdict(dict)
    for f in glob.glob(pattern):
        for r in csv.DictReader(open(f)):
            try:
                d[int(r["epoch"])][int(r["seed"])] = (float(r[pcol]), float(r[extra])) if extra else float(r[pcol])
            except (KeyError, ValueError):
                continue
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="")
    a = ap.parse_args()
    files = sorted(glob.glob(f"{B}/chat_stats/chatstats_*_c*.csv"))
    all_tags = {os.path.basename(f)[len("chatstats_"):].rsplit("_c", 1)[0] for f in files}       # zoo_c10_m2_c15-36 -> zoo_c10_m2
    tags = sorted(all_tags)
    if a.tags:
        tags = [t for t in tags if t in a.tags.split(",")]
    for tag in tags:
        p = collections.defaultdict(lambda: collections.defaultdict(dict))      # stat -> epoch -> seed -> p
        obs = collections.defaultdict(dict)                                     # epoch -> seed -> median difference
        val = collections.defaultdict(lambda: collections.defaultdict(dict))    # summary -> epoch -> seed -> [tr, te]
        dup = conf = unparsable = 0
        for f in sorted(glob.glob(f"{B}/chat_stats/chatstats_{tag}_c*.csv")):     # sorted: the later file wins on duplicates
            for r in csv.DictReader(open(f)):
                if r.get("tag") != tag:
                    if r.get("tag") in all_tags:                                   # rows of another model in this file
                        raise SystemExit(f"{f}: tag column {r['tag']} does not match the file name tag {tag}")
                    unparsable += 1; continue                                     # torn line from concurrent writers
                if None in r:                                                     # more fields than the header
                    unparsable += 1; continue
                try:
                    e, s, name = int(r["epoch"]), int(r["seed"]), r["stat"]
                    o = float(r["obs"])
                    pv = float(r["p"]) if name in STATS else None
                except (TypeError, ValueError):
                    unparsable += 1; continue
                if name in STATS:
                    if s in p[name][e]:
                        dup += 1
                        conf += abs(p[name][e][s] - pv) > 1e-9 or (name == "med" and abs(obs[e][s] - o) > 1e-9)
                    p[name][e][s] = pv
                    if name == "med":
                        obs[e][s] = o
                elif name.startswith(("tr_", "te_")):
                    side, key = name.split("_", 1)
                    slot = val[key][e].setdefault(s, [None, None]); i = 0 if side == "tr" else 1
                    if slot[i] is not None:
                        dup += 1
                        conf += abs(slot[i] - o) > 1e-9
                    slot[i] = o
        all_eps = sorted(p["med"])
        complete = [e for e in all_eps if set(p["med"][e]) == SEEDS and all(set(p[n][e]) == SEEDS for n in STATS)]
        incomplete = [e for e in all_eps if e not in complete]
        if not complete:
            print(f"\n== {tag}: no epoch with the complete seeds 42 to 91 yet ({len(all_eps)} epochs started)"); continue
        vrt = read_seed_p(f"{B}/vrt_fwd/vrt_fwd_{tag}_large_c*.csv", "p", "dc")
        missing = [(e, s) for e in complete for s in SEEDS if s not in vrt.get(e, {})]
        pairs = [(e, s) for e in complete for s in SEEDS if s in vrt.get(e, {})]
        bad_p = sum(abs(p["med"][e][s] - vrt[e][s][0]) > 1e-6 for e, s in pairs)
        bad_dc = sum(abs(obs[e][s] - vrt[e][s][1]) > 1e-5 for e, s in pairs)
        vrt_med = {e: st.median(v[0] for v in vrt[e].values()) for e in vrt if len(vrt[e]) >= 50}
        chat = read_seed_p(f"{B}/mmd_on_chat/mmdchat_{tag}_c*.csv")
        chat_med = {e: st.median(chat[e].values()) for e in chat if len(chat[e]) >= 50}
        medp = {name: {e: st.median(p[name][e].values()) for e in complete} for name in STATS}
        print(f"\n== {tag}: evaluated epochs {complete[0]} to {complete[-1]}, {len(complete)} complete (seeds 42 to 91)"
              + (f", incomplete {incomplete}" if incomplete else ""))
        print(f"   check against vrt_fwd: {len(pairs)} of {len(complete) * 50} pairs found, {len(missing)} missing, "
              f"{bad_dc} median-difference mismatches, {bad_p} p mismatches")
        print(f"   rows: {dup} duplicate (epoch, seed, stat) rows, {conf} of them with a different value, {unparsable} unparsable"
              + ("" if dup or unparsable else " (clean)"))
        print(f"   whole-run onsets: VRT {onset(vrt_med)}, MMD on c-hat {onset(chat_med)} "
              f"(vrt_fwd {len(vrt_med)} epochs, mmd_on_chat {len(chat_med)} epochs with 50 seeds)")
        print("   onset inside the window: " + "  ".join(f"{name} {onset(medp[name])}" for name in STATS))
        print("epoch " + " ".join(f"{name:>6s}" for name in STATS) + "  mmd_c |  te - tr, median over seeds: " + " ".join(f"{k:>7s}" for k in SUMS))
        for e in complete:
            diffs = []
            for k in SUMS:
                d = [v[1] - v[0] for v in val[k][e].values() if None not in v]
                diffs.append(st.median(d) if d else float("nan"))
            mc = f"{chat_med[e]:6.3f}" if e in chat_med else "     -"
            print(f"{e:5d} " + " ".join(f"{medp[name][e]:6.3f}" for name in STATS) + f" {mc} |  " + " ".join(f"{d:+7.4f}" for d in diffs))


if __name__ == "__main__":
    main()
