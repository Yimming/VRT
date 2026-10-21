"""Validated SimCLR loader and shared panel drawing; final build applies 0.07/0.5 loss conversion."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import sys
import numpy as np

OUT = Path(__file__).resolve().parent
from plot_loss_representatives import load, onsets, audit, plt, BLUE, RED, GREEN
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

SOURCE = OUT / 'data'
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8,
                     'axes.titlesize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'savefig.dpi': 360,
                     'path.simplify': False})
ssl_audit = {}
source_hashes = {}


def rows(path):
    source_hashes[str(path.relative_to(OUT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def load_ssl(ds, expected_onset):
    loss_files = sorted((SOURCE / 'ssl_nll').glob(f'ssl_nll_{ds}_c*.csv'))
    records = [r for path in loss_files for r in rows(path)]
    records.sort(key=lambda r: int(r['epoch']))
    a = np.array([[float(r[k]) for k in ('epoch', 'tr_eval', 'te_eval', 'gap_eval')]
                  for r in records])
    assert len(loss_files) == 10
    assert np.array_equal(a[:, 0], np.arange(1, 1001)), ds
    assert np.isfinite(a).all() and (a[:, 1:3] >= 0).all()
    assert np.max(np.abs((a[:, 2] - a[:, 1]) - a[:, 3])) < 1.01e-6
    vals = defaultdict(dict)
    for path in sorted((SOURCE / 'ssl').glob(f'ssl_{ds}_fast_c*.csv')):
        for r in rows(path):
            if r['stat'] != 'vrt' or r['budget'] != 'b10000':
                continue
            ep, seed, p = int(r['epoch']), int(r['seed']), float(r['p'])
            assert seed not in vals[ep], (path, ep, seed)
            vals[ep][seed] = p
    assert sorted(vals) == list(range(1, 1001)), ds
    assert all(set(v) == set(range(42, 92)) for v in vals.values())
    epochs = np.array(sorted(vals))
    median_p = np.array([np.median(list(vals[ep].values())) for ep in epochs])
    assert np.isfinite(median_p).all() and ((median_p > 0) & (median_p <= 1)).all()
    sustained = np.flatnonzero(np.logical_and.accumulate((median_p < .05)[::-1])[::-1])
    onset = int(epochs[sustained[0]]) if len(sustained) else None
    assert onset == expected_onset, (ds, onset, expected_onset)
    ssl_audit[ds] = {
        'n_loss_checkpoints': len(a), 'displayed_epoch_range': [1, 1000],
        'BN_mode': 'eval', 'fields': ['tr_eval', 'te_eval'],
        'scale': 'Original CSV SupConLoss units, no conversion',
        'vrt_onset_recomputed': onset, 'seeds_per_vrt_checkpoint': 50,
        'loss_at_onset': a[a[:, 0] == onset].tolist()[0],
        'last_epoch_loss': a[-1].tolist(),
        'test_minimum_epoch': int(a[np.argmin(a[:, 2]), 0]),
        'test_minimum': float(a[:, 2].min()),
        'note': 'Fixed views and batches; not an average of 50 training runs.'
    }
    return a[:, :3], onset


def draw_panel(ax, a, onset, model, dataset, letter, ssl=False, ylabel=False):
    ep, tr, te = a.T
    ax.plot(ep, tr, color=BLUE, lw=.9, label='Train')
    ax.plot(ep, te, color=RED, lw=.9, ls='--', label='Test')
    if onset is not None:
        ax.axvline(onset, color=GREEN, lw=.8, ls='--', zorder=0)
    note = f'VRT {onset}' if onset is not None else 'No VRT onset'
    ax.text(.97, .95, note, transform=ax.transAxes, ha='right', va='top',
            color=GREEN if onset is not None else '#555555', fontsize=7,
            bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.9, 'pad':.5})
    end = 200 if model in ('ResNet20-FRN', 'ResNet-50') else ep[-1]
    ax.set_xlim(0, end)
    ax.set_xticks([0, 500, 1000] if end == 1000 else [0, 100, 200] if end == 200
                 else [0, 30, 60] if end == 60 else [0, 25, 50])
    if ssl:
        # Common scale for both SimCLR datasets; show the full loss range.
        ax.set_ylim(35, 50)
        ax.set_yticks([35, 40, 45, 50])
        ax.set_ylabel('Contrastive loss', labelpad=4)
    else:
        ax.set_ylim(0, max(tr.max(), te.max()) * 1.22)
        ax.yaxis.set_major_locator(MaxNLocator(3, min_n_ticks=3))
        if ylabel:
            ax.set_ylabel('Cross-entropy loss', labelpad=4)
    ax.tick_params(pad=2)
    ax.set_title(f'({letter}) {model}\n{dataset}', loc='left', pad=6, linespacing=1.2)
    ax.set_xlabel('Epoch', labelpad=3)

