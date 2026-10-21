"""harvest_ftg8.py - merged every-epoch onsets for the big-budget table.
Unions ALL csv sources per (model, stat, budget, space) after rejoining
chunks (r50_c0..c3 -> r50, v7a/b -> v7). Paper onset rule throughout."""
import csv, glob, os, re
from collections import defaultdict
from statistics import median

ROOT = '/scratch/p318770/model_zoos/ladder_out'

def norm_model(m):
    m = m.lower()
    m = re.sub(r'_c\d+$', '', m)
    return 'v7' if m in ('v7a', 'v7b') else m

vrt = defaultdict(lambda: defaultdict(list))
base = defaultdict(lambda: defaultdict(list))
accs = defaultdict(lambda: defaultdict(list))
for f in glob.glob(ROOT + '/**/*.csv', recursive=True):
    b = os.path.basename(f)
    if 'ranks' in b or b.startswith('bn_'): continue
    stem = re.sub(r'\.csv$', '', b)
    with open(f) as fh:
        rd = csv.reader(fh); hdr = next(rd, None)
        if not hdr: continue
        idx = {c.strip().lower(): i for i, c in enumerate(hdr)}
        if 'p' not in idx or 'epoch' not in idx: continue
        for row in rd:
            if not row or row[0] == hdr[0]: continue
            try:
                ep = int(row[idx['epoch']]); pv = float(row[idx['p']])
            except (ValueError, IndexError): continue
            st = row[idx['stat']].lower() if 'stat' in idx else ''
            bud = row[idx['budget']].lower() if 'budget' in idx else ''
            sp = row[idx['space']].lower() if 'space' in idx else ''
            m = norm_model(row[idx['model']]) if 'model' in idx else ''
            if (st == 'chat' and bud == 'paper' and stem.startswith('ladder_')
                    and '_bntrain' not in stem and '_seed' not in stem
                    and 'phase' not in stem):
                vrt[m][ep].append(pv)
            if bud in ('b5000', 'b8000', 'b10k'):
                base[(m, st, bud, sp)][ep].append(pv)
                if st == 'c2st' and 'acc' in idx:
                    try: accs[(m, bud)][ep].append(float(row[idx['acc']]))
                    except ValueError: pass

def onset(pm):
    vals = [v for l in pm.values() for v in l]
    flag = set(round(v, 10) for v in vals) <= {0.0, 1.0}
    dec = {e: (sum(l)/len(l) >= 0.5 if flag else median(l) < 0.05)
           for e, l in pm.items()}
    o = None
    for e in sorted(dec, reverse=True):
        if dec[e]: o = e
        else: break
    return o

MODELS = ['r50', 'v7', 'abla', 'ablb', 'ablc', 'abld',
          'zoo_c10_m0', 'zoo_c10_m1', 'zoo_c10_m2', 'zoo_c10_m3', 'zoo_c10_m4',
          'zoo_c100_m0', 'zoo_c100_m1', 'zoo_c100_m2', 'zoo_c100_m3',
          'zoo_c100_m4', 'wp']
COLS = [('c2st', 'b8000', ''), ('c2st', 'b10k', ''), ('ed', 'b10k', ''),
        ('mmd', 'b10k', 'l2'), ('mmd', 'b10k', 'unnorm'), ('mmdagg', 'b5000', '')]
print(f'{"model":13s} {"VRT":>4s}  {"c2st8k":>10s} {"c2st10k":>14s} {"ed10k":>7s} '
      f'{"mmd_l2":>7s} {"mmd_un":>7s} {"mmdagg":>7s}')
for m in MODELS:
    o_v = onset(vrt[m]) if m in vrt else None
    cells = []
    for st, bud, sp in COLS:
        pm = base.get((m, st, bud, sp))
        if not pm:
            cells.append('n/a'); continue
        o = onset(pm)
        cell = 'ND' if o is None else str(o)
        if st == 'c2st' and o is not None and o in accs.get((m, bud), {}):
            cell += f'({median(accs[(m, bud)][o]):.3f})'
        cells.append(cell)
    print(f'{m:13s} {("ND" if o_v is None else str(o_v)):>4s}  '
          f'{cells[0]:>10s} {cells[1]:>14s} {cells[2]:>7s} '
          f'{cells[3]:>7s} {cells[4]:>7s} {cells[5]:>7s}')
