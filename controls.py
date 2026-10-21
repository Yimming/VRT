"""Random-network, split and extraction controls (Appendix C)."""
import argparse
import copy
import csv
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from common import l2n, parse_range
from vrt import vrt_full, vrt_low
from models import (ARCHS, N_TR, N_TE, build, set_bn, load_cifar,
                    extract_batch as extract, DeepResNet50,
                    ImageNetTrain, ImageNetVal)

BUDGETS = {"large": dict(k=60, M=2000, n_qtr=8000),
           "small": dict(k=15, M=500, n_qtr=1000, n_qte=1500)}
N_TIER, N_AGG = 10000, 5000
PARTS = {"gpu": ["mmd", "ed", "henze", "schilling"], "cpu": ["c2st", "mmdagg"]}
BUDGET = {"mmd": N_TIER, "ed": N_TIER, "henze": N_TIER, "schilling": N_TIER,
          "c2st": N_TIER, "mmdagg": N_AGG}
METHODS = ["mmd", "ed", "vrt"]


def _l2n32(x):
    return l2n(x.astype(np.float32))


def cifar_vrt(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=list(ARCHS), required=True)
    ap.add_argument('--bn_mode', choices=['eval', 'train'], default='eval')
    ap.add_argument('--reps', required=True, help='repetitions, e.g. 0-99')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--n_perm', type=int, default=2000)
    ap.add_argument('--seed0', type=int, default=1000, help='network and split seed of repetition r: seed0 + r')
    ap.add_argument('--vrt_seed0', type=int, default=100000, help='VRT seed of repetition r: vrt_seed0 + r')
    ap.add_argument('--dry_run', action='store_true', help='random images instead of CIFAR (pipeline test, file tagged DRYRUN)')
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args(argv)
    if args.arch == 'frn_c10' and args.bn_mode == 'train':
        raise SystemExit('ResNet20-FRN has no BatchNorm: train mode would repeat eval mode; run eval only')
    reps = parse_range(args.reps)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bs = ARCHS[args.arch][2]
    tag = f'{args.arch}_{args.bn_mode}' + ('_DRYRUN' if args.dry_run else '')
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f'vrt_null_{tag}_r{reps[0]}-{reps[-1]}.csv')
    done = {}
    for f in glob.glob(os.path.join(args.out_dir, f'vrt_null_{tag}_r*.csv')):
        if os.path.getsize(f) == 0:
            continue
        for r in csv.DictReader(open(f)):
            done.setdefault(int(r['rep']), set()).add((r['split'], r['budget']))
    full = {(s, b) for s in ('resplit', 'official') for b in BUDGETS}
    todo = [r for r in reps if not full <= done.get(r, set())]
    gpu = torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'
    print(f'{tag}: {len(todo)} of {len(reps)} repetitions to do ({reps[0]}-{reps[-1]}), device {gpu}, {args.n_perm} permutations -> {out_csv}', flush=True)
    if not todo:
        return
    X = load_cifar(args.arch, args.dry_run, data_root=args.data_root)
    if device == 'cuda':
        X = X.to(device)
    print(f'images {tuple(X.shape)}  normalized mean {X.mean().item():+.3f}  std {X.std().item():.3f}', flush=True)
    order_all = np.arange(N_TR + N_TE)
    official = (order_all[:N_TR], order_all[N_TR:])
    new = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    fout = open(out_csv, 'a', newline='')
    w = csv.writer(fout)
    if new:
        w.writerow(['arch', 'bn_mode', 'rep', 'net_seed', 'split_seed', 'vrt_seed', 'split', 'budget', 'D', 'dc', 'p', 'cos', 'feat_sec'])
        fout.flush()
    rej = {(s, b): [0, 0] for s in ('resplit', 'official') for b in BUDGETS}
    for r in todo:
        t0 = time.time()
        net_seed, split_seed, vrt_seed = (args.seed0 + r, args.seed0 + r, args.vrt_seed0 + r)
        torch.manual_seed(net_seed)
        model, fwd = build(args.arch)
        model = model.to(device)
        perm = np.random.default_rng(split_seed).permutation(N_TR + N_TE)
        resplit = (perm[N_TE:], perm[:N_TE])
        assert len(np.intersect1d(resplit[0], resplit[1])) == 0
        if args.bn_mode == 'eval':
            set_bn(model, 'eval')
            h = extract(model, fwd, X, order_all, bs, device)
            H = {'resplit': (h[resplit[0]], h[resplit[1]]), 'official': (h[:N_TR], h[N_TR:])}
        else:
            sd0 = copy.deepcopy(model.state_dict())
            H = {}
            for name, (a, b) in (('resplit', resplit), ('official', official)):
                parts = []
                for order in (a, b):
                    model.load_state_dict(sd0)
                    set_bn(model, 'train')
                    parts.append(extract(model, fwd, X, order, bs, device))
                H[name] = tuple(parts)
        t_feat = time.time() - t0
        rows, msg, coss = ([], [], {})
        for split in ('resplit', 'official'):
            h_tr, h_te = H[split]
            finite = bool(np.isfinite(h_tr).all() and np.isfinite(h_te).all())
            e_tr, e_te = (l2n(h_tr.astype(np.float64)), l2n(h_te.astype(np.float64)))
            D = e_tr.shape[1]
            cos = coss[split] = float((e_tr[:1000] @ e_te[:1000].T).mean()) if finite else float('nan')
            ps = []
            for budget, kw in BUDGETS.items():
                if not finite:
                    dc, p = (float('nan'), float('nan'))
                elif budget == 'large':
                    dc, p = vrt_full(e_tr, e_te, D, n_perm=args.n_perm, seed=vrt_seed, **kw)
                else:
                    dc, p = vrt_low(e_tr, e_te, D, n_perm=args.n_perm, seed=vrt_seed, **kw)
                rows.append([args.arch, args.bn_mode, r, net_seed, split_seed, vrt_seed, split, budget, D, f'{dc:+.6e}', f'{p:.6f}', f'{cos:.4f}', f'{t_feat:.1f}'])
                if finite:
                    rej[split, budget][0] += p < 0.05
                    rej[split, budget][1] += 1
                ps.append(p)
            msg.append(f'{split} p {ps[0]:.3f} / {ps[1]:.3f}' + ('' if finite else ' NON-FINITE FEATURES'))
        w.writerows(rows)
        fout.flush()
        hn = np.median(np.linalg.norm(H['resplit'][0][:2000], axis=1))
        print(f"rep {r:4d}  D {D}  |h| median {hn:.3g}  cos {coss['resplit']:.3f}  {'  '.join(msg)}  (features {t_feat:.0f} s, total {time.time() - t0:.0f} s)", flush=True)
        del model, H
    fout.close()
    print('this task, rejections at 0.05 (large / small):  ' + '  '.join((f"{s} {rej[s, 'large'][0]}/{rej[s, 'large'][1]} , {rej[s, 'small'][0]}/{rej[s, 'small'][1]}" for s in ('resplit', 'official'))), flush=True)

def run_gpu_methods(e_tr, e_te, seed, n_perm, methods):
    from baselines import mmd_test_fast
    from baselines import energy_test_fast
    from baselines import knn_graph, schilling_p, henze_p
    out = {}
    n = min(N_TIER, len(e_tr), len(e_te))
    if 'mmd' in methods:
        out['mmd'] = (mmd_test_fast(e_tr, e_te, N_TIER, n_perm, seed, 'torch')[1], np.nan)
    if 'ed' in methods:
        out['ed'] = (energy_test_fast(e_tr, e_te, n, n_perm, seed, 'torch')[1], np.nan)
    if 'henze' in methods or 'schilling' in methods:
        rng = np.random.default_rng(seed)
        X = e_tr[rng.choice(len(e_tr), n, replace=False)]
        Y = e_te[rng.choice(len(e_te), n, replace=False)]
        Z = np.vstack([X, Y])
        tags = np.array([0] * n + [1] * n)
        idx = knn_graph(Z, 3, 'torch')
        out['schilling'] = (schilling_p(idx, tags, n, n, 3), np.nan)
        out['henze'] = (henze_p(idx, tags, n_perm, seed), np.nan)
    return out

def cifar_baselines(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=list(PARTS), required=True)
    ap.add_argument('--methods', default=None, help="subset of the part's methods, comma separated (default: all)")
    ap.add_argument('--arch', choices=list(ARCHS), required=True)
    ap.add_argument('--bn_mode', choices=['eval'], default='eval')
    ap.add_argument('--reps', required=True)
    ap.add_argument('--splits', default=None, help='resplit,official (default: both for gpu, resplit for cpu)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--n_perm', type=int, default=2000)
    ap.add_argument('--seed0', type=int, default=1000)
    ap.add_argument('--vrt_seed0', type=int, default=100000)
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args(argv)
    methods = args.methods.split(',') if args.methods else PARTS[args.part]
    assert set(methods) <= set(PARTS[args.part]), f'{methods} not in part {args.part}'
    splits = (args.splits or ('resplit,official' if args.part == 'gpu' else 'resplit')).split(',')
    reps = parse_range(args.reps)
    if args.part == 'gpu':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device != 'cuda' and (not args.dry_run):
            raise SystemExit('--part gpu needs a GPU (the s50 MMD/ED/Henze numbers used the torch backend on the GPU)')
    else:
        device = 'cpu'
        if 'mmdagg' in methods:
            from baselines import mmdagg_fn
            if mmdagg_fn is None:
                raise SystemExit('mmdagg not importable: load the module first, then activate the venv')
    tag = f'{args.arch}_{args.bn_mode}_{args.part}' + ('_DRYRUN' if args.dry_run else '')
    os.makedirs(args.out_dir, exist_ok=True)
    extra = '' if args.splits is None else '_' + '-'.join(splits)
    out_csv = os.path.join(args.out_dir, f'base_null_{tag}_r{reps[0]}-{reps[-1]}{extra}.csv')
    done = {}
    for f in glob.glob(os.path.join(args.out_dir, f'base_null_{tag}_r*.csv')):
        if os.path.getsize(f) == 0:
            continue
        for r in csv.DictReader(open(f)):
            done.setdefault(int(r['rep']), set()).add((r['split'], r['method']))
    need = {(s, m) for s in splits for m in methods}
    todo = [r for r in reps if not need <= done.get(r, set())]
    print(f"{tag}: methods {methods}, splits {splits}: {len(todo)} of {len(reps)} repetitions to do ({reps[0]}-{reps[-1]}), device {(torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu')}, {torch.get_num_threads()} threads -> {out_csv}", flush=True)
    if not todo:
        return
    X = load_cifar(args.arch, args.dry_run, data_root=args.data_root)
    if device == 'cuda':
        X = X.to(device)
    bs = ARCHS[args.arch][2]
    official = (np.arange(N_TR), np.arange(N_TR, N_TR + N_TE))
    new = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    fout = open(out_csv, 'a', newline='')
    w = csv.writer(fout)
    if new:
        w.writerow(['arch', 'bn_mode', 'rep', 'net_seed', 'split_seed', 'seed', 'split', 'method', 'budget', 'p', 'acc', 'sec'])
        fout.flush()
    rej = {(s, m): [0, 0] for s in splits for m in methods}
    for r in todo:
        t0 = time.time()
        net_seed, split_seed, seed = (args.seed0 + r, args.seed0 + r, args.vrt_seed0 + r)
        torch.manual_seed(net_seed)
        model, fwd = build(args.arch)
        model = model.to(device)
        set_bn(model, 'eval')
        perm = np.random.default_rng(split_seed).permutation(N_TR + N_TE)
        resplit = (perm[N_TE:], perm[:N_TE])
        pairs = {'resplit': resplit, 'official': official}
        rows, msg = ([], [])
        if args.part == 'gpu':
            l2n = _l2n32
            h = extract(model, fwd, X, np.arange(N_TR + N_TE), bs, device)
            t_feat = time.time() - t0
            for s in splits:
                tr, te = pairs[s]
                e_tr, e_te = (l2n(h[tr]), l2n(h[te]))
                t1 = time.time()
                res = run_gpu_methods(e_tr, e_te, seed, args.n_perm, methods)
                for m in methods:
                    p, acc = res[m]
                    rows.append([args.arch, args.bn_mode, r, net_seed, split_seed, seed, s, m, BUDGET[m], f'{p:.6f}', 'nan', f'{time.time() - t1:.0f}'])
                    rej[s, m][0] += p < 0.05
                    rej[s, m][1] += 1
                msg.append(s + ' ' + ' '.join((f'{m} {res[m][0]:.3f}' for m in methods)))
        else:
            from baselines import c2st_test, mmdagg_test
            l2n = _l2n32
            t_feat = 0.0
            for s in splits:
                tr, te = pairs[s]
                pos = np.random.RandomState(seed).permutation(len(tr))[:N_TIER]
                t1 = time.time()
                hf = extract(model, fwd, X, np.concatenate([tr[pos], te]), bs, device)
                t_feat += time.time() - t1
                D = hf.shape[1]
                e_tr = np.full((len(tr), D), np.nan, np.float32)
                e_tr[pos] = hf[:len(pos)]
                with np.errstate(invalid='ignore'):
                    e_tr, e_te = (l2n(e_tr), l2n(hf[len(pos):]))
                assert np.all(np.linalg.norm(e_tr[pos], axis=1) > 0.5)
                for m in methods:
                    t1 = time.time()
                    if m == 'c2st':
                        acc, p = c2st_test(e_tr, e_te, n_sub=N_TIER, seed=seed)
                    else:
                        acc, p = (np.nan, float(mmdagg_test(e_tr, e_te, n_sub=N_AGG, seed=seed)))
                    rows.append([args.arch, args.bn_mode, r, net_seed, split_seed, seed, s, m, BUDGET[m], f'{p:.6f}', f'{acc:.6f}' if m == 'c2st' else 'nan', f'{time.time() - t1:.0f}'])
                    hit = p >= 0.5 if m == 'mmdagg' else p < 0.05
                    rej[s, m][0] += hit
                    rej[s, m][1] += 1
                    msg.append(f'{s} {m} ' + (f'p {p:.3f} acc {acc:.3f}' if m == 'c2st' else f'reject {int(p)}'))
        w.writerows(rows)
        fout.flush()
        print(f"rep {r:4d}  {'  '.join(msg)}  (features {t_feat:.0f} s, total {time.time() - t0:.0f} s)", flush=True)
        del model
    fout.close()
    print('this task, single-test rejections at 0.05:  ' + '  '.join((f'{s} {m} {k}/{n}' for (s, m), (k, n) in rej.items())), flush=True)

def cifar_rule(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=list(ARCHS), required=True)
    ap.add_argument('--nets', required=True, help='network indices, e.g. 0-9')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--net_seed0', type=int, default=5000, help='network i: torch.manual_seed(net_seed0 + i)')
    ap.add_argument('--seeds', default='42-91', help="analysis seeds (the paper's: 42-91)")
    ap.add_argument('--budgets', default='large,small')
    ap.add_argument('--n_perm', type=int, default=2000)
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args(argv)
    nets, seeds, budgets = (parse_range(args.nets), parse_range(args.seeds), args.budgets.split(','))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda' and (not args.dry_run):
        raise SystemExit('needs a GPU for the feature extraction')
    bs = ARCHS[args.arch][2]
    tag = f'{args.arch}_eval' + ('_DRYRUN' if args.dry_run else '')
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f'vrt_rule_{tag}_n{nets[0]}-{nets[-1]}.csv')
    done = {}
    for f in glob.glob(os.path.join(args.out_dir, f'vrt_rule_{tag}_n*.csv')):
        if os.path.getsize(f) == 0:
            continue
        for r in csv.DictReader(open(f)):
            done.setdefault(int(r['net']), set()).add((r['budget'], int(r['seed'])))
    need = {(b, s) for b in budgets for s in seeds}
    todo = [i for i in nets if not need <= done.get(i, set())]
    print(f"{tag}: {len(todo)} of {len(nets)} networks to do ({nets[0]}-{nets[-1]}), official split, seeds {seeds[0]}-{seeds[-1]}, budgets {budgets}, device {(torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu')} -> {out_csv}", flush=True)
    if not todo:
        return
    X = load_cifar(args.arch, args.dry_run, data_root=args.data_root)
    if device == 'cuda':
        X = X.to(device)
    new = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    fout = open(out_csv, 'a', newline='')
    w = csv.writer(fout)
    if new:
        w.writerow(['arch', 'bn_mode', 'net', 'net_seed', 'budget', 'seed', 'dc', 'p'])
        fout.flush()
    flagged = {b: [0, 0] for b in budgets}
    for i in todo:
        t0 = time.time()
        torch.manual_seed(args.net_seed0 + i)
        model, fwd = build(args.arch)
        model = model.to(device)
        set_bn(model, 'eval')
        h = extract(model, fwd, X, np.arange(N_TR + N_TE), bs, device)
        h_tr, h_te = (h[:N_TR], h[N_TR:])
        t_feat = time.time() - t0
        if not (np.isfinite(h_tr).all() and np.isfinite(h_te).all()):
            print(f'net {i}: NON-FINITE FEATURES, skipped', flush=True)
            continue
        e64_tr, e64_te = (l2n(h_tr.astype(np.float64)), l2n(h_te.astype(np.float64)))
        if args.arch.startswith('simclr'):
            e_lg_tr, e_lg_te = (l2n(h_tr).astype(np.float32), l2n(h_te).astype(np.float32))
        else:
            e_lg_tr, e_lg_te = (e64_tr, e64_te)
        D = h.shape[1]
        rows, msg = ([], [])
        for b in budgets:
            ps = []
            for s in seeds:
                if b == 'large':
                    dc, p = vrt_full(e_lg_tr, e_lg_te, D, k=60, M=2000, n_qtr=8000, n_perm=args.n_perm, seed=s)
                else:
                    dc, p = vrt_low(e64_tr, e64_te, D, k=15, M=500, n_qtr=1000, n_qte=1500, n_perm=args.n_perm, seed=s)
                rows.append([args.arch, 'eval', i, args.net_seed0 + i, b, s, f'{dc:+.6e}', f'{p:.6f}'])
                ps.append(p)
            med = float(np.median(ps))
            hit = med < 0.05
            flagged[b][0] += hit
            flagged[b][1] += 1
            msg.append(f'{b}: median p {med:.3f}, {sum((p < 0.05 for p in ps))}/{len(ps)} seeds < 0.05' + ('  FLAGGED' if hit else ''))
        w.writerows(rows)
        fout.flush()
        print(f'net {i:3d}  ' + '   '.join(msg) + f'  (features {t_feat:.0f} s, total {time.time() - t0:.0f} s)', flush=True)
        del model
    fout.close()
    print("this task, networks flagged by the paper's rule:  " + '  '.join((f'{b} {k}/{n}' for b, (k, n) in flagged.items())), flush=True)

def cifar_official(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', choices=['pixels', 'r18_shuffled'], required=True)
    ap.add_argument('--dataset', choices=['cifar10', 'cifar100'], required=True)
    ap.add_argument('--reps', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--n_perm', type=int, default=2000)
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args(argv)
    from baselines import mmd_test_fast
    l2n32 = _l2n32
    from baselines import energy_test_fast
    from vrt import vrt_full
    l2n64 = l2n
    reps = parse_range(args.reps)
    arch = 'r18_c10' if args.dataset == 'cifar10' else 'r18_c100'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tag = f'{args.features}_{args.dataset}' + ('_DRYRUN' if args.dry_run else '')
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f'official_check_{tag}_r{reps[0]}-{reps[-1]}.csv')
    done = {}
    for f in glob.glob(os.path.join(args.out_dir, f'official_check_{tag}_r*.csv')):
        if os.path.getsize(f) > 0:
            for r in csv.DictReader(open(f)):
                done.setdefault(int(r['rep']), set()).add((r['split'], r['method']))
    need = {(s, m) for s in ('official', 'resplit') for m in METHODS}
    todo = [r for r in reps if not need <= done.get(r, set())]
    print(f'{tag}: {len(todo)} of {len(reps)} repetitions to do, device {device}, {torch.get_num_threads()} threads -> {out_csv}', flush=True)
    if not todo:
        return
    X = load_cifar(arch, args.dry_run, data_root=args.data_root)
    if args.features == 'r18_shuffled' and device == 'cuda':
        X = X.to(device)
    official = (np.arange(N_TR), np.arange(N_TR, N_TR + N_TE))
    new = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    fout = open(out_csv, 'a', newline='')
    w = csv.writer(fout)
    if new:
        w.writerow(['dataset', 'features', 'rep', 'split', 'method', 'p'])
        fout.flush()
    for r in todo:
        t0 = time.time()
        seed = 100000 + r
        if args.features == 'pixels':
            h = X.reshape(len(X), -1).numpy()
        else:
            torch.manual_seed(1000 + r)
            model, fwd = build(arch)
            model = model.to(device)
            set_bn(model, 'eval')
            order = np.random.default_rng(7000000 + r).permutation(N_TR + N_TE)
            h_shuf = extract(model, fwd, X, order, 512, device)
            h = np.empty_like(h_shuf)
            h[order] = h_shuf
            del model
        perm = np.random.default_rng(1000 + r).permutation(N_TR + N_TE)
        pairs = {'official': official, 'resplit': (perm[N_TE:], perm[:N_TE])}
        rows, msg = ([], [])
        for s in ('official', 'resplit'):
            tr, te = pairs[s]
            e_tr, e_te = (l2n32(h[tr]), l2n32(h[te]))
            n = min(10000, len(e_tr), len(e_te))
            p_mmd = mmd_test_fast(e_tr, e_te, 10000, args.n_perm, seed, 'torch')[1]
            p_ed = energy_test_fast(e_tr, e_te, n, args.n_perm, seed, 'torch')[1]
            f_tr, f_te = (l2n64(h[tr].astype(np.float64)), l2n64(h[te].astype(np.float64)))
            p_vrt = vrt_full(f_tr, f_te, f_tr.shape[1], k=60, M=2000, n_qtr=8000, n_perm=args.n_perm, seed=seed)[1]
            for m, p in (('mmd', p_mmd), ('ed', p_ed), ('vrt', p_vrt)):
                rows.append([args.dataset, args.features, r, s, m, f'{p:.6f}'])
            msg.append(f'{s} mmd {p_mmd:.3f} ed {p_ed:.3f} vrt {p_vrt:.3f}')
        w.writerows(rows)
        fout.flush()
        print(f'rep {r:4d}  ' + '  '.join(msg) + f'  ({time.time() - t0:.0f} s)', flush=True)
    fout.close()


@torch.inference_mode()
def _imagenet_normalized_features(model, loader, max_n, data_seed, device):
    torch.manual_seed(data_seed)
    model.eval()
    out, count = [], 0
    for x, _ in loader:
        features = F.normalize(model.forward_features(x.to(device)), dim=1).cpu()
        out.append(features)
        count += features.shape[0]
        if count >= max_n:
            break
    return torch.cat(out, 0)[:max_n].numpy().astype(np.float64)


def imagenet_vrt(argv=None):
    parser = argparse.ArgumentParser(description="ImageNet random-network control with the original preprocessing choices.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preproc", choices=["paper", "cluster"], default="paper")
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--seed0", type=int, default=0)
    args = parser.parse_args(argv)
    from vrt import vrt_test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.preproc == "paper":
        transform = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])
    else:
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                               T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_ds = ImageNetTrain(os.path.join(args.data_root, "train"), transform)
    test_ds = ImageNetVal(os.path.join(args.data_root, "val"), transform)
    if not test_ds.paths:
        test_ds.paths = sorted(glob.glob(os.path.join(args.data_root, "val", "*", "*.JPEG")))
    kw = dict(batch_size=256, num_workers=4, pin_memory=True)
    tr_loader = DataLoader(train_ds, shuffle=True, **kw)
    te_loader = DataLoader(test_ds, shuffle=True, **kw)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = set()
    if os.path.isfile(args.out):
        with open(args.out) as source:
            done = {int(row["seed"]) for row in csv.DictReader(source)}
    new = not os.path.isfile(args.out) or os.path.getsize(args.out) == 0
    with open(args.out, "a", newline="") as target:
        writer = csv.writer(target)
        if new:
            writer.writerow(["seed", "dc", "p"])
        for seed in range(args.seed0, args.seed0 + args.n_seeds):
            if seed in done:
                continue
            torch.manual_seed(seed)
            model = DeepResNet50().to(device)
            e_tr = _imagenet_normalized_features(model, tr_loader, 50000, 42, device)
            e_te = _imagenet_normalized_features(model, te_loader, 10000, 123, device)
            dc, p = vrt_test(e_tr, e_te, D=2048, seed=seed)
            writer.writerow([seed, f"{dc:.6f}", f"{p:.6f}"])
            target.flush()
            print(f"seed {seed}: dc={dc:.6f}, p={p:.6f}", flush=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    commands = {
        "cifar-vrt": (cifar_vrt, "VRT on fresh networks; random and official image splits"),
        "cifar-baselines": (cifar_baselines, "MMD, ED, kNN, C2ST and MMDAgg on fresh networks"),
        "cifar-rule": (cifar_rule, "50-seed VRT decision rule on fresh networks"),
        "cifar-official": (cifar_official, "Pixel and shuffled-extraction split controls"),
        "imagenet-vrt": (imagenet_vrt, "Original ImageNet random-network VRT control"),
    }
    parser = argparse.ArgumentParser(description="Calibration and split controls (Appendix C). For ImageNet random or disjoint feature files, use models.py --random-init / --pair trtr.")
    sub = parser.add_subparsers(dest="control", required=True)
    for name, (_, description) in commands.items():
        sub.add_parser(name, help=description, add_help=False)
    if argv and argv[0] in commands:
        return commands[argv[0]][0](argv[1:])
    parser.parse_args(argv)


if __name__ == "__main__":
    main()
