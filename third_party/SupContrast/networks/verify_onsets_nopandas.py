#!/usr/bin/env python3
"""verify_ladder_onsets, pandas-free (stdlib csv only). Same logic/expectations."""
import argparse, csv, glob, os, re, sys
from collections import defaultdict
from statistics import median

EXP = [
 (['r50','norm','paper'],61),(['r50','cos','paper'],119),
 (['r50','theta','paper'],59),(['r50','l_k','paper'],59),
 (['r50','chat','paper'],61),
 (['abla','chat'],31),(['ablc','chat'],35),(['v7','chat'],19),
 (['ablb','chat'],None),(['ablb','norm'],44),(['ablb','cos'],45),(['ablb','theta'],45),
 (['abld','chat'],None),(['abld','norm'],48),(['abld','cos'],46),(['abld','theta'],46),
 (['wp','chat'],None),(['wp','theta'],None),(['wp','norm'],None),(['wp','cos'],None),
 (['zoo_c100','cos'],None),(['zoo_c100','chat'],(22,34)),(['zoo_c100','theta'],(20,27)),
 (['zoo_c10_','chat'],(22,28)),(['seed','chat'],61),
 (['r50','mmd','l2','b1500'],121),(['r50','mmd','l2','b10k'],80),
 (['r50','mmd','unnorm','b1500'],115),(['r50','mmd','unnorm','b10k'],65),
 (['r50','c2st','b1500'],131),(['r50','c2st','b8000'],106),(['r50','c2st','b10k'],65),
 (['r50','ed','b1500'],121),(['r50','ed','b10k'],80),
 (['r50','mmdagg','b1500'],121),(['r50','mmdagg','b5000'],65),
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

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', action='append', required=True)
    args = ap.parse_args()
    files = []
    for r in args.root:
        for pat in ('ladder_*.csv','adapt_*.csv','unnorm_*.csv','base*_*.csv','seed_*.csv'):
            files += glob.glob(os.path.join(r,'**',pat), recursive=True)
    files = sorted(set(f for f in files if 'ranks' not in f))
    print(f'# {len(files)} files')
    groups = defaultdict(lambda: defaultdict(list))
    for f in files:
        with open(f) as fh:
            rd = csv.reader(fh); hdr = next(rd, None)
            if not hdr: continue
            idx = {c.strip().lower(): i for i,c in enumerate(hdr)}
            if 'p' not in idx or 'epoch' not in idx:
                print('  SKIP', f, hdr); continue
            stem = re.sub(r'\.csv$','',os.path.basename(f))
            for row in rd:
                if not row or row[0]==hdr[0]: continue
                try:
                    ep = int(row[idx['epoch']]); p = float(row[idx['p']])
                except (ValueError, IndexError): continue
                model = row[idx['model']] if 'model' in idx else stem
                stat = row[idx['stat']] if 'stat' in idx else 'stat?'
                bud = row[idx['budget']] if 'budget' in idx else ''
                sp = row[idx['space']] if 'space' in idx else ''
                key = f'{stem}|{model}|{stat}|{bud}|{sp}'.lower()
                groups[key][ep].append(p)
    res = {}
    print('\n== onsets ==')
    for key in sorted(groups):
        vals = [v for eps in groups[key].values() for v in eps]
        is_flag = set(round(v,10) for v in vals) <= {0.0, 1.0}
        o = onset(groups[key], is_flag)
        res[key] = o
        print(f'{key:70s} onset={o} {"FLAG" if is_flag else ""} rows={len(vals)}')
    print('\n== checks ==')
    ok=bad=miss=0
    for pats, exp in EXP:
        hits = [(k,v) for k,v in res.items() if all(p in k for p in pats)]
        if not hits: print(f'  ?  {pats} not found'); miss+=1; continue
        for k,got in hits:
            if exp is None: good = got is None; want='null'
            elif isinstance(exp, tuple): good = got is not None and exp[0]<=got<=exp[1]; want=f'{exp}'
            else: good = got==exp; want=str(exp)
            print(f'  {"OK      " if good else "MISMATCH"} {k}: got {got}, want {want}')
            ok+=good; bad+=not good
    print(f'\nSummary: {ok} OK, {bad} MISMATCH, {miss} not-found')

if __name__ == '__main__':
    main()
