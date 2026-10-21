"""
s50_make_tasks.py — build the task lists for the 50-seed rerun (seeds 42-91) of every method on every
feature set, writing to a fresh results root (default benchmark_final/s50). Nothing under the old
directories is touched. Run on the cluster from /scratch/p318770/ssl_supcon_code_s50:

    python s50_make_tasks.py            # scans the feature folders, writes s50_gpu_tasks.txt / s50_cpu_tasks.txt
                                        # and prints the two sbatch commands with the right --array ranges

Every line of a task file is one shell command for the array drivers job_s50_gpu.sh / job_s50_cpu.sh
(task i runs line i+1). Long settings are cut into epoch chunks so that no task exceeds the targets below;
each chunk writes its own CSV (suffix _c<lo>-<hi>), so chunks never share a file. All scripts resume, so a
failed task can simply be resubmitted with the same array index.

Feature sets (all /scratch/p318770/model_zoos/ladder_feats/<tag>/ep{ep:03d}.npz):
  eval-mode : r50 phase_bs8 phase_bs128 phase_bs1024 v7 ablA ablB ablC ablD wp zoo_c10_m0-4 zoo_c100_m0-4
              imgnet_simple imgnet_adv
  train-mode: <tag>_bntrain for r50, ablA-D, the ten zoo models, imgnet_simple, imgnet_adv (skipped if absent)
Groups (per-epoch costs for 50 seeds, measured on the 10-seed runs and scaled):
  GPU  base   : MMD and ED, normalized and raw, 1,500 and 10,000; Schilling/Henze normalized, both budgets
  CPU  vrt    : forward VRT large (paper protocol) and small (k=15), reversed VRT large and small (eval-mode
                sets only; train-mode sets get the forward tests)
  CPU  c2st   : C2ST at 1,500 and 10,000, every epoch
  CPU  agg1500  (every epoch), agg5000  (every 5th epoch; every epoch on ImageNet and phase)
SimCLR (features extracted on the fly) is a separate pair of task families produced by the same script:
  GPU  ssl fast (VRT large and small, MMD, ED and C2ST at both budgets, every epoch 1-1000)
  CPU  ssl slow (MMDAgg 1,500, every 5th epoch)
"""
import os, glob, re, argparse, math, time

ap = argparse.ArgumentParser()
ap.add_argument("--code", default="/scratch/p318770/ssl_supcon_code_s50", help="code folder; task files and s50_logs/ are written here")
ap.add_argument("--feats", default="/scratch/p318770/model_zoos/ladder_feats")
ap.add_argument("--ssl_ckpts", default="/scratch/p318770/model_zoos/ssl_supcon")
ap.add_argument("--bf", default="/home3/p318770/venvs/overfitting/benchmark_final/s50")
ap.add_argument("--seeds", default="42-91")
ap.add_argument("--gpu_hours", type=float, default=5.0, help="target wall time per GPU task")
ap.add_argument("--cpu_hours", type=float, default=7.0, help="target wall time per CPU task")
ap.add_argument("--cpu_hours_long", type=float, default=9.0, help="target for MMDAgg 5,000 tasks (one evaluated epoch is 4.2 h)")
ap.add_argument("--no_ssl", action="store_true")
ap.add_argument("--no_bntrain", action="store_true")
args = ap.parse_args()
os.chdir(args.code)                      # the drivers cd here and read the task files relative to it


def parse_range(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


NS = len(parse_range(args.seeds))
EVAL = (["r50", "phase_bs8", "phase_bs128", "phase_bs1024", "v7", "ablA", "ablB", "ablC", "ablD", "wp"]
        + [f"zoo_c10_m{i}" for i in range(5)] + [f"zoo_c100_m{i}" for i in range(5)] + ["imgnet_simple", "imgnet_adv"])
TRAIN = ([f"{t}_bntrain" for t in ["r50", "ablA", "ablB", "ablC", "ablD"]]
         + [f"zoo_c10_m{i}_bntrain" for i in range(5)] + [f"zoo_c100_m{i}_bntrain" for i in range(5)]
         + ["imgnet_simple_bntrain", "imgnet_adv_bntrain"])
SPARSE = {"imgnet_simple", "imgnet_adv", "imgnet_simple_bntrain", "imgnet_adv_bntrain",
          "phase_bs8", "phase_bs128", "phase_bs1024"}          # few checkpoints: slow tests on every one

# seconds per seed per evaluated epoch (10-seed timings), by group
COST = {"base": 1.5 + 1.5 + 0.3 + 0.3 + 1.5 + 1.5 + 0.3 + 0.3 + 1.5 + 0.3,   # mmd n/r x2 budgets, ed n/r x2, knn x2
        "vrt_eval": 5 + 1 + 5 + 1, "vrt_train": 5 + 1,
        "c2st1500": 1.5, "c2st10k": 5, "agg1500": 30, "agg5000": 300,
        "ssl_fast": 5 + 1 + 1.5 + 1.5 + 0.3 + 0.3 + 1.5 + 5, "ssl_slow": 30}
EXTRACT = {"ssl_fast": 90, "ssl_slow": 600}      # feature extraction per epoch (GPU / CPU)


def epochs_of(tag):
    out = []
    for f in glob.glob(os.path.join(args.feats, tag, "ep*.npz")):
        m = re.fullmatch(r"ep(\d+)\.npz", os.path.basename(f))
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def chunks(eps, per_epoch_s, target_h, fixed_s=0.0):
    """Cut the evaluated-epoch list into consecutive chunks whose estimated time stays under target."""
    out, cur, t = [], [], fixed_s
    for e in eps:
        if cur and t + per_epoch_s > target_h * 3600:
            out.append(cur); cur, t = [], fixed_s
        cur.append(e); t += per_epoch_s
    if cur:
        out.append(cur)
    return out


gpu, cpu = [], []
hours = {"gpu": 0.0, "cpu": 0.0}
BF, F, S = args.bf, args.feats, args.seeds
sub = ["mmd_p2000", "mmd_raw_p2000", "ed_p2000", "ed_raw_p2000", "knn_v2", "vrt_fwd", "reversed", "c2st", "mmdagg", "ssl"]
for d in sub:
    os.makedirs(os.path.join(BF, d), exist_ok=True)
os.makedirs("s50_logs", exist_ok=True)

sets = [(t, "eval") for t in EVAL] + ([] if args.no_bntrain else [(t, "train") for t in TRAIN])
for tag, mode in sets:
    eps = epochs_of(tag)
    if not eps:
        print(f"WARNING: no features for {tag}, skipped"); continue
    span = lambda ch: f"{ch[0]}-{ch[-1]}"
    npz = f"{F}/{tag}/ep{{ep:03d}}.npz"
    # ---- GPU: MMD, ED (normalized + raw), kNN at both budgets
    for ch in chunks(eps, NS * COST["base"], args.gpu_hours):
        r, c = span(ch), f"_c{span(ch)}"; cmds = []
        for N in (1500, 10000):
            for scr, d, flag in (("mmd_tst_fast.py", "mmd_p2000", ""), ("mmd_tst_fast.py", "mmd_raw_p2000", " --raw"),
                                 ("energy_tst_fast.py", "ed_p2000", ""), ("energy_tst_fast.py", "ed_raw_p2000", " --raw")):
                pre = "mmd" if scr.startswith("mmd") else "ed"
                cmds.append(f"python {scr} --tag {tag} --npz_fmt '{npz}' --n_samples {N} --n_perm 2000 --seeds {S} --epochs {r} --backend torch{flag} --out {BF}/{d}/{pre}_{tag}_n{N}{c}.csv")
            cmds.append(f"python knn_tst_budget.py --tag {tag} --npz_fmt '{npz}' --n_samples {N} --k 3 --n_perm 2000 --seeds {S} --epochs {r} --backend torch --out {BF}/knn_v2/knn_{tag}_n{N}_part{c}.csv")
        gpu.append(" && ".join(cmds)); hours["gpu"] += len(ch) * NS * COST["base"] / 3600
    # ---- CPU: VRT forward (both budgets) and, for eval-mode sets, reversed (both budgets)
    key = "vrt_eval" if mode == "eval" else "vrt_train"
    for ch in chunks(eps, NS * COST[key], args.cpu_hours):
        r, c = span(ch), f"_c{span(ch)}"
        cmds = [f"python vrt_forward_budget.py --tag {tag} --budget {b} --npz_fmt '{npz}' --epochs {r} --seeds {S} --n_perm 2000 --out {BF}/vrt_fwd/vrt_fwd_{tag}_{b}{c}.csv" for b in ("large", "small")]
        if mode == "eval":
            cmds += [f"python vrt_reversed_budget.py --tag {tag} --budget {b} --npz_fmt '{npz}' --epochs {r} --seeds {S} --n_perm 2000 --out {BF}/reversed/vrt_rev_{tag}_{b}{c}.csv" for b in ("large", "small")]
        cpu.append(" && ".join(cmds)); hours["cpu"] += len(ch) * NS * COST[key] / 3600
    if mode == "train":
        continue                                   # C2ST and MMDAgg only on the eval-mode (Table 1) features
    every5 = 1 if tag in SPARSE else 5
    e5 = [e for e in eps if every5 == 1 or e % 5 == 0]
    # ---- CPU: C2ST at 1,500 and 10,000, every epoch (one call per chunk, both budgets)
    for ch in chunks(eps, NS * (COST["c2st1500"] + COST["c2st10k"]), args.cpu_hours):
        r = span(ch)
        cpu.append(f"python base_fill.py --stat c2st --tag {tag} --npz_fmt '{npz}' --budgets 1500,10000 --every 1 --seeds {S} --epochs {r} --out_dir {BF}/c2st --out_suffix _c{r}")
        hours["cpu"] += len(ch) * NS * (COST["c2st1500"] + COST["c2st10k"]) / 3600
    # ---- CPU: MMDAgg 1,500 every epoch; 5,000 every 5th
    for ch in chunks(eps, NS * COST["agg1500"], args.cpu_hours):
        r = span(ch)
        cpu.append(f"python base_fill.py --stat mmdagg --tag {tag} --npz_fmt '{npz}' --budgets 1500 --every 1 --seeds {S} --epochs {r} --out_dir {BF}/mmdagg --out_suffix _c{r}")
        hours["cpu"] += len(ch) * NS * COST["agg1500"] / 3600
    for ch in chunks(e5, NS * COST["agg5000"], args.cpu_hours_long):
        r = span(ch)
        cpu.append(f"python base_fill.py --stat mmdagg --tag {tag} --npz_fmt '{npz}' --budgets 5000 --every {every5} --seeds {S} --epochs {r} --out_dir {BF}/mmdagg --out_suffix _c{r}")
        hours["cpu"] += len(ch) * NS * COST["agg5000"] / 3600

# ---- SimCLR: features on the fly
if not args.no_ssl:
    for ds in ("cifar10", "cifar100"):
        cks = sorted(glob.glob(os.path.join(args.ssl_ckpts, f"SimCLR_{ds}_*")))
        if not cks:
            print(f"WARNING: no SimCLR checkpoint folder for {ds}, skipped"); continue
        ck = cks[0]
        eps = sorted(int(re.search(r"ckpt_epoch_(\d+)\.pth", os.path.basename(f)).group(1))
                     for f in glob.glob(os.path.join(ck, "ckpt_epoch_*.pth")))
        eps = [e for e in eps if 1 <= e <= 1000]
        for ch in chunks(eps, NS * COST["ssl_fast"] + EXTRACT["ssl_fast"], args.gpu_hours):
            gpu.append(f"python ssl_s50_fill.py --ckpt_dir {ck} --dataset {ds} --group fast --lo {ch[0]} --hi {ch[-1]} --seeds {S} --out_dir {BF}/ssl")
            hours["gpu"] += len(ch) * (NS * COST["ssl_fast"] + EXTRACT["ssl_fast"]) / 3600
        e5 = [e for e in eps if e % 5 == 0]
        for ch in chunks(e5, NS * COST["ssl_slow"] + EXTRACT["ssl_slow"], args.cpu_hours):
            cpu.append(f"python ssl_s50_fill.py --ckpt_dir {ck} --dataset {ds} --group slow --every 5 --lo {ch[0]} --hi {ch[-1]} --seeds {S} --out_dir {BF}/ssl")
            hours["cpu"] += len(ch) * (NS * COST["ssl_slow"] + EXTRACT["ssl_slow"]) / 3600

# task files carry a time stamp: array indices refer to ONE file, never regenerate and reuse old indices
stamp = time.strftime("%Y%m%d_%H%M")
MAXARRAY = 1000                                  # Slurm MaxArraySize is often 1001; split larger families
print(f"GPU tasks: {len(gpu)}  (about {hours['gpu']:.0f} GPU-hours)")
print(f"CPU tasks: {len(cpu)}  (about {hours['cpu']:.0f} CPU-task-hours)")
print("submit with (from the code folder):")
for fam, lines, drv, conc in (("gpu", gpu, "job_s50_gpu.sh", 60), ("cpu", cpu, "job_s50_cpu.sh", 120)):
    for part, start in enumerate(range(0, len(lines), MAXARRAY)):
        chunk = lines[start:start + MAXARRAY]
        name = f"s50_{fam}_tasks_{stamp}" + (f"_part{part + 1}" if len(lines) > MAXARRAY else "") + ".txt"
        open(name, "w").write("\n".join(chunk) + "\n")
        print(f"  sbatch --array=0-{len(chunk) - 1}%{conc} {drv} {name}")
        if part == 0:                        # short tasks for a smoke test before the full submission
            smoke = [i for i, l in enumerate(chunk) if (" --tag imgnet_simple " in l and "vrt_forward" not in l and "base_fill" not in l)
                     or (" --tag phase_bs8 " in l and ("vrt_forward" in l or "stat c2st" in l))]
            if smoke:
                print(f"    smoke test (minutes): sbatch --array={','.join(map(str, smoke[:2]))} {drv} {name}")
