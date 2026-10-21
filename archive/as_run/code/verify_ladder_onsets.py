#!/usr/bin/env python3
"""
verify_ladder_onsets.py — C3 final confirmation over ladder/baseline CSVs
=========================================================================
Run ON THE CLUSTER, then paste the full stdout back for final confirmation:

    python3 verify_ladder_onsets.py --root /scratch/p318770/model_zoos/ladder_out
    # add more roots if baselines/unnorm live elsewhere:
    python3 verify_ladder_onsets.py --root $OUT --root /home3/p318770/venvs/overfitting/benchmark_final

What it does
------------
1. Recursively globs CSVs matching --pattern (default: ladder_*.csv,
   adapt_*.csv, unnorm_*.csv, baselines2_*.csv, seed_*.csv).
2. Expects columns: epoch, p (+ any of: model, stat/method, budget, seed,
   space). Missing model column -> inferred from the file name.
3. Auto-detects whether `p` is a p-value or a 0/1 reject flag per group
   (the C1 lesson — some outputs store binary decisions):
      all values in {0,1}  -> flag column; epoch rejects if mean over
                              seeds >= 0.5   (WARNING printed if only 1.0s
                              appear — ambiguous, check manually)
      otherwise            -> p-values; epoch rejects if median over
                              seeds < 0.05
4. Sustained onset per (model, stat, budget, space) = first available epoch
   from which every subsequent available epoch rejects (paper convention).
5. Prints the full onset table, then checks the built-in EXPECTATIONS
   (Table 1 + ledger + today's confirmations) as OK / MISMATCH / not-found.
   The full table is the ground truth; the expectation block is convenience.
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd
import numpy as np

DEFAULT_PATTERNS = ['ladder_*.csv', 'adapt_*.csv', 'unnorm_*.csv',
                    'baselines2_*.csv', 'seed_*.csv']

# ── expectations ─────────────────────────────────────────────────────────
# (list of substring-patterns that must ALL appear in "model|stat|budget|space",
#  expected)  expected: int exact | (lo, hi) inclusive range | None = null.
# A pattern like '1500|b1500' matches if ANY alternative is a substring.
EXPECTATIONS = [
    # r50 ladder, paper budget
    (['r50', 'norm', 'paper'], 61),
    (['r50', 'cos', 'paper'], 119),
    (['r50', 'theta', 'paper'], 59),
    (['r50', 'l_k|lk|logvol', 'paper'], 59),
    (['r50', 'chat|c_hat', 'paper'], 61),
    # pipeline reconciliation targets
    (['ablA|abl_a', 'chat|c_hat'], 31),
    (['ablC|abl_c', 'chat|c_hat'], 35),
    (['v7', 'chat|c_hat'], 19),
    # ablB / ablD (today's numbers)
    (['ablB|abl_b', 'chat|c_hat'], None),
    (['ablB|abl_b', 'norm'], 44),
    (['ablB|abl_b', 'cos'], 45),
    (['ablB|abl_b', 'theta'], 45),
    (['ablD|abl_d', 'chat|c_hat'], None),
    (['ablD|abl_d', 'norm'], 48),
    (['ablD|abl_d', 'cos'], 46),
    (['ablD|abl_d', 'theta'], 46),
    # WildPark: everything silent
    (['wp', 'chat|c_hat'], None),
    (['wp', 'theta'], None),
    (['wp', 'norm'], None),
    (['wp', 'cos'], None),
    # zoo claims (checked per matching model, e.g. each zoo_c100_m*)
    (['zoo_c100', 'cos'], None),          # <- key rebuttal claim: all five null
    (['zoo_c100', 'chat|c_hat'], (22, 34)),
    (['zoo_c100', 'theta'], (20, 27)),
    (['zoo_c10', 'chat|c_hat'], (22, 28)),
    # r50 seeds
    (['seed', 'chat|c_hat'], 61),
    # baselines on r50 (only if those CSVs are under a --root)
    (['r50', 'mmd$|mmd_l2|^mmd', 'l2', '1500'], 121),
    (['r50', 'mmd$|mmd_l2|^mmd', 'l2', '10k|10000'], 80),
    (['r50', 'mmd', 'unnorm|raw', '1500'], 115),
    (['r50', 'mmd', 'unnorm|raw', '10k|10000'], 65),
    (['r50', 'c2st', '1500'], 131),
    (['r50', 'c2st', '8000'], 106),
    (['r50', 'c2st', '10k|10000'], 65),
    (['r50', 'ed|energy', '1500'], 121),
    (['r50', 'ed|energy', '10k|10000'], 80),
    (['r50', 'mmdagg|agg', '1500'], 121),
    (['r50', 'mmdagg|agg', '5000'], 65),
]

ALPHA = 0.05


def sustained_onset(g, is_flag):
    per = g.groupby('epoch')['p']
    dec = (per.mean() >= 0.5) if is_flag else (per.median() < ALPHA)
    dec = dec.sort_index()
    onset = None
    for ep in reversed(dec.index.tolist()):
        if dec.loc[ep]:
            onset = ep
        else:
            break
    return onset


def detect_flag(vals):
    u = np.unique(np.round(vals.astype(float), 12))
    if set(u).issubset({0.0, 1.0}):
        return True, (set(u) == {1.0})
    return False, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', action='append', required=True)
    ap.add_argument('--pattern', action='append', default=None)
    args = ap.parse_args()
    patterns = args.pattern or DEFAULT_PATTERNS

    files = []
    for root in args.root:
        for pat in patterns:
            files += glob.glob(os.path.join(root, '**', pat), recursive=True)
    files = sorted(set(files))
    if not files:
        sys.exit('no CSVs found — check --root/--pattern')
    print(f'# {len(files)} CSV files')

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, engine='python')
        except Exception as e:
            print(f'  SKIP {f}: {e}')
            continue
        cols = {c.lower(): c for c in df.columns}
        if 'p' not in cols or 'epoch' not in cols:
            print(f'  SKIP {f}: no p/epoch columns ({list(df.columns)})')
            continue
        df = df.rename(columns={cols['p']: 'p', cols['epoch']: 'epoch'})
        statcol = cols.get('stat') or cols.get('method')
        df['stat'] = df[statcol].astype(str) if statcol else 'stat?'
        if 'model' in cols:
            df['model'] = df[cols['model']].astype(str)
        else:
            stem = re.sub(r'\.csv$', '', os.path.basename(f))
            df['model'] = re.sub(r'^(ladder|adapt|unnorm|baselines2|seed)_?',
                                 '', stem) or stem
        df['budget'] = df[cols['budget']].astype(str) if 'budget' in cols else ''
        df['space'] = df[cols['space']].astype(str) if 'space' in cols else ''
        frames.append(df[['model', 'stat', 'budget', 'space', 'epoch', 'p']])
    data = pd.concat(frames, ignore_index=True)

    # ── onset table ──
    results = []
    for (m, s, b, sp), g in data.groupby(['model', 'stat', 'budget', 'space']):
        is_flag, ambiguous = detect_flag(g['p'].values)
        onset = sustained_onset(g, is_flag)
        note = 'FLAG-col' if is_flag else ''
        if ambiguous:
            note += ' AMBIGUOUS(all-1.0: reject-flag or p=1.0? check!)'
        results.append((m, s, b, sp, onset, len(g), note))
    res = pd.DataFrame(results, columns=['model', 'stat', 'budget', 'space',
                                         'onset', 'n_rows', 'note'])
    res = res.sort_values(['model', 'budget', 'space', 'stat'])
    print('\n== sustained onsets (None = null) ==')
    print(res.to_string(index=False))

    # ── expectation checks ──
    def match(keystr, pats):
        return all(any(a in keystr for a in p.split('|')) for p in pats)

    print('\n== expectation checks ==')
    n_ok = n_bad = n_missing = 0
    for pats, exp in EXPECTATIONS:
        hits = [r for r in res.itertuples()
                if match(f'{r.model}|{r.stat}|{r.budget}|{r.space}'.lower(),
                         [p.lower() for p in pats])]
        if not hits:
            print(f'  ?  {pats}: not found in these roots')
            n_missing += 1
            continue
        for r in hits:
            got = r.onset
            if exp is None:
                ok = got is None
                want = 'null'
            elif isinstance(exp, tuple):
                ok = got is not None and exp[0] <= got <= exp[1]
                want = f'{exp[0]}-{exp[1]}'
            else:
                ok = got == exp
                want = str(exp)
            tag = 'OK      ' if ok else 'MISMATCH'
            if ok:
                n_ok += 1
            else:
                n_bad += 1
            print(f'  {tag} {r.model}|{r.stat}|{r.budget}|{r.space}: '
                  f'got {got}, expected {want} {r.note}')
    print(f'\nSummary: {n_ok} OK, {n_bad} MISMATCH, {n_missing} not-found')
    print('Paste this whole output back for final confirmation.')


if __name__ == '__main__':
    main()
