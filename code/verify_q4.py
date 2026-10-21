"""verify_q4.py - recompute every cell of nxR4 Table 3 from raw CSVs.
Sustained-onset rule (identical to all prior harvests): per epoch take the
median p across analysis seeds (0/1 flag columns use mean>=0.5), onset =
first epoch from which the decision stays reject through the final
checkpoint. Prints every relevant group first, then targeted checks."""
import csv, glob, os, re, sys
from collections import defaultdict
from statistics import median

ROOT = '/scratch/p318770/model_zoos/ladder_out'
SHOW = ('mmd', 'chat', 'mean', 'unnorm')   # print groups whose key mentions these

EXP = [
 ('Table3 a  raw MMD r50 = 115',  [['r50','unnorm','1500']], 115),
 ('Table3 a  L2 MMD r50 = 121',   [['r50','mmd','l2','1500'],['r50','l2','1500']], 121),
 ('Table3 a  raw MMD v7 = nd',    [['v7','unnorm','1500']], None),
 ('Table3 a  L2 MMD v7 = nd',     [['v7','mmd','l2','1500'],['v7','l2','1500']], None),
 ('Table3 b  MMD-on-scores r50 = 75', [['mmdchat','r50'],['mmd_chat','r50'],['r50','mmd','chat']], 75),
 ('Table3 b  MMD-on-scores v7 = nd',  [['mmdchat','v7'],['mmd_chat','v7'],['v7','mmd','chat']], None),
 ('Table3 c  VRT r50 = 61',       [['ladder_r50','|chat|','paper'],['ladder_r50','|chat|'],['r50','chat','paper']], 61),
 ('Table3 c  VRT v7 = 19',        [['ladder_v7','|chat|'],['v7','chat']], 19),
 ('Row4 mean r50 = 61',           [['meandiff','r50'],['mean','r50']], 61),
 ('Row4 mean v7 = 24',            [['meandiff','v7'],['mean','v7']], 24),
]

def onset(pmap, is_flag):
    eps = sorted(pmap)
    dec = {e: (sum(pmap[e])/len(pmap[e]) >= 0.5) if is_flag
              else (median(pmap[e]) < 0.05) for e in eps}
    o = None
    for e in reversed(eps):
        if dec[e]: o = e
        else: break
    return o

files = sorted(set(glob.glob(os.path.join(ROOT, '**', '*.csv'), recursive=True)))
files = [f for f in files if 'ranks' not in f and 'bn_' not in os.path.basename(f)]
print(f'# scanning {len(files)} csv files under {ROOT}')
groups = defaultdict(lambda: defaultdict(list))
for f in files:
    with open(f) as fh:
        rd = csv.reader(fh); hdr = next(rd, None)
        if not hdr: continue
        idx = {c.strip().lower(): i for i, c in enumerate(hdr)}
        if 'p' not in idx or 'epoch' not in idx: continue
        stem = re.sub(r'\.csv$', '', os.path.basename(f))
        for row in rd:
            if not row or row[0] == hdr[0]: continue
            try:
                ep = int(row[idx['epoch']]); pv = float(row[idx['p']])
            except (ValueError, IndexError): continue
            model = row[idx['model']] if 'model' in idx else stem
            stat  = row[idx['stat']]  if 'stat'  in idx else 'stat?'
            bud   = row[idx['budget']] if 'budget' in idx else ''
            sp    = row[idx['space']]  if 'space'  in idx else ''
            key = f'{stem}|{model}|{stat}|{bud}|{sp}'.lower()
            groups[key][ep].append(pv)

res = {}
print('\n== relevant groups ==')
for key in sorted(groups):
    vals = [v for eps in groups[key].values() for v in eps]
    is_flag = set(round(v, 10) for v in vals) <= {0.0, 1.0}
    o = onset(groups[key], is_flag)
    res[key] = o
    if any(t in key for t in SHOW):
        print(f'{key:78s} onset={o} rows={len(vals)}')

print('\n== Table 3 checks ==')
ok = bad = miss = 0
for label, patsets, exp in EXP:
    hits = []
    for pats in patsets:
        hits = [(k, v) for k, v in res.items() if all(p in k for p in pats)]
        if hits: break
    if not hits:
        print(f'  ?        {label}: no matching group'); miss += 1; continue
    for k, got in hits:
        good = (got is None) if exp is None else (got == exp)
        print(f'  {"OK      " if good else "MISMATCH"} {label}: got {got}   [{k}]')
        ok += good; bad += not good
print(f'\nSummary: {ok} OK, {bad} MISMATCH, {miss} not-found')
