"""
plot_dc_trajectory_s50.py — Figure 8: dc trajectories under train-as-reference (Standard) and test-as-reference
(Reversed), v7, r50 and the two ImageNet models (camera-ready).
dc values are medians over the 50 seeds (42 to 91) at 10,000 points per split, read from fig_data_s50/dc_<tag>.csv
(written by export_fig_data.py from the s50 forward and reversed VRT files). The submitted figure had hard-coded arrays:
its Standard curves were the medians over seeds 42 to 51 of the same forward VRT runs; its Reversed curves came from an
earlier reversed run that the s50 files do not reproduce. Onset lines from fig_data_s50/onsets.csv (v7 38, r50 61).
Same style template as the other figures; global legend; no per-panel legends.
"""
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.join(ROOT, 'fig_data_s50')
OUT = ROOT

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.labelsize': 7,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.4,
    'ytick.major.width': 0.4,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'lines.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

C_N = '#2166AC'   # Standard / normal direction
C_R = '#D6191B'   # Reversed direction
C_ZERO = '#CCCCCC'
C_ONSET = '#AAAAAA'


def load_dc(tag):
    with open(os.path.join(NEW, f'dc_{tag}.csv')) as f:
        rows = list(csv.DictReader(f))
    num = lambda s: float(s) if s not in ('', None) else np.nan
    return (np.array([int(r['epoch']) for r in rows]),
            np.array([num(r['dc_standard']) for r in rows]),
            np.array([num(r['dc_reversed']) for r in rows]))


def load_onsets():
    with open(os.path.join(NEW, 'onsets.csv')) as f:
        return {r['tag']: (int(r['vrt_onset']) if r['vrt_onset'] else None) for r in csv.DictReader(f)}


ONSET = load_onsets()
v7_eps, v7_dc_N, v7_dc_R = load_dc('v7')
r50_eps, r50_dc_N, r50_dc_R = load_dc('r50')
img_eps, img_s_dc_N, img_s_dc_R = load_dc('imgnet_simple')
img_a_eps, img_a_dc_N, img_a_dc_R = load_dc('imgnet_adv')
print(f"v7 {len(v7_eps)} epochs (onset {ONSET['v7']}), r50 {len(r50_eps)} epochs (onset {ONSET['r50']}), "
      f"ImageNet checkpoints {list(img_eps)}")

# ── Plot ──
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.85))

# (a) v7
ax = axes[0]
ax.plot(v7_eps, v7_dc_N, color=C_N, linewidth=0.6, alpha=0.85)
ax.plot(v7_eps, v7_dc_R, color=C_R, linewidth=0.6, alpha=0.85)
ax.axhline(0, color=C_ZERO, lw=0.4, zorder=0)
ax.axvline(ONSET['v7'], color=C_ONSET, ls='--', lw=0.4)
ax.set_xlabel('Epoch')
ax.set_ylabel('dc')
ax.set_title('(a) v7 ($D$=64)', fontsize=7, loc='left')
ax.set_xlim(0, 199)

# (b) r50
ax = axes[1]
ax.plot(r50_eps, r50_dc_N, color=C_N, linewidth=0.6, alpha=0.85)
ax.plot(r50_eps, r50_dc_R, color=C_R, linewidth=0.6, alpha=0.85)
ax.axhline(0, color=C_ZERO, lw=0.4, zorder=0)
ax.axvline(ONSET['r50'], color=C_ONSET, ls='--', lw=0.4)
ax.set_xlabel('Epoch')
ax.set_title('(b) r50 ($D$=2048)', fontsize=7, loc='left')
ax.set_xlim(0, 200)

# (c) ImageNet simple
ax = axes[2]
ax.plot(img_eps, img_s_dc_N, color=C_N, marker='o', markersize=2.5, linewidth=0.8)
ax.plot(img_eps, img_s_dc_R, color=C_R, marker='s', markersize=2.5, linewidth=0.8)
ax.axhline(0, color=C_ZERO, lw=0.4, zorder=0)
ax.set_xlabel('Epoch')
ax.set_title('(c) ImageNet simple', fontsize=7, loc='left')
ax.set_xlim(0, 310)

# (d) ImageNet advanced
ax = axes[3]
ax.plot(img_a_eps, img_a_dc_N, color=C_N, marker='o', markersize=2.5, linewidth=0.8)
ax.plot(img_a_eps, img_a_dc_R, color=C_R, marker='s', markersize=2.5, linewidth=0.8)
ax.axhline(0, color=C_ZERO, lw=0.4, zorder=0)
ax.set_xlabel('Epoch')
ax.set_title('(d) ImageNet advanced', fontsize=7, loc='left')
ax.set_xlim(0, 310)

# Global legend
legend_handles = [
    Line2D([0], [0], color=C_N, lw=0.8, label='Standard'),
    Line2D([0], [0], color=C_R, lw=0.8, label='Reversed'),
]
fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False,
           handlelength=1.6, columnspacing=1.2, handletextpad=0.4, fontsize=5.8)
plt.subplots_adjust(left=0.06, right=0.985, bottom=0.25, top=0.78, wspace=0.38)

fig.savefig(os.path.join(OUT, 'fig_dc_trajectory.pdf'))
fig.savefig(os.path.join(OUT, 'fig_dc_trajectory.png'))
print(f"Saved to {OUT}")
plt.show()
