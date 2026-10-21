"""gap_check.py - audit the every-epoch grid for the big-budget baselines.
Unions epochs across ALL csv stems (base2_/unnorm_/fill_/bigfill_ dirs) per
(model, stat, budget, space) after normalizing chunk names (r50_c0->r50,
v7a/v7b->v7), then reports missing epochs vs the expected window."""
import csv, glob, os, re
from collections import defaultdict

ROOT = '/scratch/p318770/model_zoos/ladder_out'
BUDGETS = {'b5000', 'b8000', 'b10k'}
WIN = {'r50': (0, 200), 'v7': (0, 199), 'abla': (0, 49), 'ablb': (0, 49),
       'ablc': (0, 49), 'abld': (0, 49),
       'zoo_c10_m0': (0, 50), 'zoo_c10_m1': (0, 50), 'zoo_c10_m2': (0, 50),
       'zoo_c10_m3': (0, 50), 'zoo_c10_m4': (0, 50),
       'zoo_c100_m0': (0, 60), 'zoo_c100_m1': (0, 60), 'zoo_c100_m2': (0, 60),
       'zoo_c100_m3': (0, 60), 'zoo_c100_m4': (0, 60)}

def norm_model(m):
    m = m.lower()
    m = re.sub(r'_c\d+$', '', m)
    if m in ('v7a', 'v7b'): m = 'v7'
    return m

def ranges(xs):
    out, s, p = [], None, None
    for x in xs:
        if s is None: s = p = x
        elif x == p + 1: p = x
        else: out.append((s, p)); s = p = x
    if s is not None: out.append((s, p))
    return ','.join(f'{a}' if a == b else f'{a}-{b}' for a, b in out)

eps = defaultdict(set)
for f in glob.glob(os.path.join(ROOT, '**', '*.csv'), recursive=True):
    b = os.path.basename(f)
    if 'ranks' in b or b.startswith('bn_'): continue
    with open(f) as fh:
        rd = csv.reader(fh); hdr = next(rd, None)
        if not hdr: continue
        idx = {c.strip().lower(): i for i, c in enumerate(hdr)}
        if 'p' not in idx or 'epoch' not in idx or 'budget' not in idx: continue
        for row in rd:
            if not row or row[0] == hdr[0]: continue
            try: ep = int(row[idx['epoch']])
            except (ValueError, IndexError): continue
            bud = row[idx['budget']].lower()
            if bud not in BUDGETS: continue
            m = norm_model(row[idx['model']] if 'model' in idx else '')
            st = row[idx['stat']].lower() if 'stat' in idx else '?'
            sp = row[idx['space']].lower() if 'space' in idx else ''
            eps[(m, st, bud, sp)].add(ep)

print(f'{"model":13s} {"stat":8s} {"budget":6s} {"space":7s} {"have":>5s} {"want":>5s}  missing')
bad = 0
for key in sorted(eps):
    m, st, bud, sp = key
    have = sorted(eps[key])
    if m in WIN:
        lo, hi = WIN[m]
        want = set(range(lo, hi + 1))
        miss = sorted(want - set(have))
        flag = 'FULL' if not miss else ranges(miss)
        if miss: bad += 1
        print(f'{m:13s} {st:8s} {bud:6s} {sp:7s} {len(have):5d} {len(want):5d}  {flag}')
    else:
        print(f'{m:13s} {st:8s} {bud:6s} {sp:7s} {len(have):5d} {"?":>5s}  window unknown, have {ranges(have)}')
print(f'\n{bad} (model,stat,budget,space) groups with gaps')
