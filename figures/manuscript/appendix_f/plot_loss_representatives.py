"""Supervised loss loader and 50-seed VRT onset checks; loss curves are single-run measurements."""
from pathlib import Path
import csv
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent / 'data'
OUT = Path(__file__).resolve().parent
BLUE, RED, GREEN = '#2166AC', '#B2182B', '#1B7837'
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8,
    'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .5, 'xtick.major.width': .45, 'ytick.major.width': .45,
    'xtick.major.size': 2.3, 'ytick.major.size': 2.3,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 240,
})


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


onsets = {r['tag']: int(r['vrt_onset']) if r['vrt_onset'] else None
          for r in read(ROOT / 'fig_data_s50/onsets.csv')}
filenames = {
    'ablA': 'losses_ablation_A_baseline.csv',
    'ablB': 'losses_ablation_B_strong_wd.csv',
    'ablC': 'losses_ablation_C_strong_aug.csv',
    'ablD': 'losses_ablation_D_wd_plus_aug.csv',
}
audit = {}


def load(tag):
    if tag in ('v7', 'r50'):
        pattern = re.compile(r'^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+[+-][\d.]+\s*$')
        lines = (ROOT / 'benchmark_final' / f'{tag}_losses.log').read_text().splitlines()
        a = np.array([m.groups() for line in lines if (m := pattern.match(line))], float)
    else:
        rows = read(ROOT / 'benchmark_final' / filenames.get(tag, f'losses_{tag}.csv'))
        a = np.array([[int(r['epoch']), float(r['tr_loss']), float(r['te_loss'])]
                      for r in rows])
    vrt = read(ROOT / 'fig_data_s50' / f'vrt_{tag}.csv')
    epochs = np.array([int(r['epoch']) for r in vrt])
    ps = np.array([float(r['vrt_p']) for r in vrt])
    assert np.array_equal(a[:, 0], epochs), tag
    assert np.isfinite(a).all() and np.all(a[:, 1:] >= 0), tag
    assert all(int(r['n_seeds']) == 50 for r in vrt), tag
    sustained = np.flatnonzero(np.logical_and.accumulate((ps < .05)[::-1])[::-1])
    onset = int(epochs[sustained[0]]) if len(sustained) else None
    assert onset == onsets[tag], (tag, onset, onsets[tag])
    audit[tag] = {
        'n_checkpoints': len(a), 'onset': onset,
        'epoch_range': [int(a[0, 0]), int(a[-1, 0])],
        'omitted_epoch0_loss': a[a[:, 0] == 0, 1:].tolist(),
        'displayed_epochs': a[a[:, 0] > 0, 0].astype(int).tolist(),
    }
    return a[a[:, 0] > 0]

