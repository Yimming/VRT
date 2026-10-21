"""
plot_fig2_s50.py — Fig 2: Hero trajectory (v7 + r50), 2x3 layout (camera-ready)

Panels:
  (a) v7 loss    (b) v7 VRT          (c) v7 C2ST accuracy
  (d) r50 loss   (e) r50 VRT + MMD   (f) r50 C2ST accuracy

Data, all medians over the 50 seeds (42 to 91) at 10,000 points per split with 2,000 permutations:
  fig_data_s50/vrt_v7.csv, vrt_r50.csv     VRT median p per epoch (written by export_fig_data.py)
  fig_data_s50/base_v7.csv, base_r50.csv   MMD median p, C2ST median p and median held-out accuracy per epoch
  fig_data_s50/onsets.csv                  sustained VRT onsets (v7 38, r50 61)
  benchmark_final/v7_losses.log, r50_losses.log   train and test cross-entropy (unchanged)
The submitted figure read v7_multiseed.log and r50_multiseed.log (10 seeds), the single-run MMD p of
benchmark_r50_cifar100_4m.csv (smallest p 0.001, so its MMD curve levelled off at 3.0) and the c2st_v7 / c2st_r50 logs.
Onsets follow the paper's rule (median p < 0.05 at every later evaluated epoch) and are computed from the plotted curves:
v7 VRT 38 (was 19); r50 VRT 61 (unchanged), MMD 73 (was 121), C2ST 121 (median p < 0.05 and median accuracy > 0.55).
The smallest p with 2,000 permutations is 1/2,001, so VRT and MMD both level off at -log10(p) = 3.3.
Put this file next to the folders fig_data_s50 and benchmark_final.
"""

import csv
import math
import os
import re

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Paths and options
# ---------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.join(ROOT, 'fig_data_s50')        # 50-seed medians and onsets
OLD = os.path.join(ROOT, 'benchmark_final')     # loss logs (unchanged)
OUT = os.path.join(os.environ.get('VRT_FIGURES_OUT', os.path.join(ROOT, '..', 'results', 'figures')), 'earlier')
os.makedirs(OUT, exist_ok=True)

# Shaded window in (e). 'MMD' shades from the VRT onset to the MMD onset (61 to 73, "12 epochs").
# 'C2ST' shades from the VRT onset to the C2ST onset (61 to 121, "60 epochs"), if Section 4.2 keeps that window.
SHADE_TO = 'MMD'

FLOOR = 5e-4      # smallest p with 2,000 permutations is 1/2,001

# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.labelsize': 7.5,
    'axes.titlesize': 8,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 5.8,
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

C_TRAIN = '#2166AC'
C_TEST = '#B2182B'
C_VRT = '#1B7837'
C_MMD = '#878787'
C_C2ST = '#D95F02'
C_ALPHA = '#BDBDBD'
C_LR = '#7570B3'
C_TXT = '#444444'
C_NULL = '#4D4D4D'


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def col(rows, key, dtype=float):
    return np.array([dtype(r[key]) if r.get(key, '') not in ('', None) else np.nan for r in rows])


def neglog10(p_arr, floor=FLOOR):
    return np.array([-math.log10(max(float(p), floor)) for p in p_arr])


def sustained_onset(epochs, reject):
    """first evaluated epoch from which every later evaluated epoch rejects"""
    reject = np.asarray(reject, dtype=bool)
    for i in range(len(epochs)):
        if reject[i:].all():
            return int(epochs[i])
    return None


def parse_loss_log(path):
    """epoch, train loss, test loss from the '  ep  tr_loss  te_loss  gap' table of the loss logs"""
    pat = re.compile(r'^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+[+-][\d.]+\s*$')
    ep, tr, te = [], [], []
    with open(path) as f:
        for line in f:
            m = pat.match(line)
            if m:
                ep.append(int(m.group(1)))
                tr.append(float(m.group(2)))
                te.append(float(m.group(3)))
    return np.array(ep), np.array(tr), np.array(te)


ONSET = {r['tag']: (int(r['vrt_onset']) if r['vrt_onset'] else None)
         for r in load_rows(os.path.join(NEW, 'onsets.csv'))}


def load_model(tag, loss_log):
    v = load_rows(os.path.join(NEW, f'vrt_{tag}.csv'))
    b = load_rows(os.path.join(NEW, f'base_{tag}.csv'))
    d = dict(vrt_ep=col(v, 'epoch', int), vrt_p=col(v, 'vrt_p'),
             base_ep=col(b, 'epoch', int), mmd_p=col(b, 'mmd_p'),
             c2st_p=col(b, 'c2st_p'), c2st_acc=col(b, 'c2st_acc'), agg=col(b, 'mmdagg_frac'))
    d['loss_ep'], d['loss_tr'], d['loss_te'] = parse_loss_log(os.path.join(OLD, loss_log))
    d['vrt_onset'] = ONSET[tag]
    if d['vrt_onset'] != sustained_onset(d['vrt_ep'], d['vrt_p'] < 0.05):
        raise SystemExit(f'{tag}: onsets.csv does not match the VRT curve')
    d['mmd_onset'] = sustained_onset(d['base_ep'], d['mmd_p'] < 0.05)
    d['c2st_onset'] = sustained_onset(d['base_ep'], (d['c2st_p'] < 0.05) & (d['c2st_acc'] > 0.55))
    ok = ~np.isnan(d['agg'])          # MMDAgg runs every 5th epoch; rule: at least 60% of seeds reject
    d['agg_onset'] = sustained_onset(d['base_ep'][ok], d['agg'][ok] >= 0.6)
    return d


def onset_line(ax, x, color):
    ax.axvline(x, color=color, ls='--', lw=0.6, alpha=0.5, zorder=1)


def lr_drops(ax, epochs):
    for e in epochs:
        ax.axvline(e, color=C_LR, ls=':', lw=0.6, alpha=0.6, zorder=1)


def alpha_line(ax, x_text):
    y = -math.log10(0.05)
    ax.axhline(y, color=C_ALPHA, ls='--', lw=0.5, zorder=0)
    ax.text(x_text, y + 0.08, r'$\alpha$=0.05', fontsize=5, color='#999999', ha='right', va='bottom')


def panel(ax, letter, title):
    ax.text(-0.30, 1.05, f'({letter})', transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='left')
    ax.set_title(title, loc='left', fontsize=7.5, pad=3)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
v7 = load_model('v7', 'v7_losses.log')
r50 = load_model('r50', 'r50_losses.log')
for name, d in (('v7', v7), ('r50', r50)):
    print(f"{name}: VRT {len(d['vrt_ep'])} epochs, onset {d['vrt_onset']}; MMD onset {d['mmd_onset']}, "
          f"C2ST onset {d['c2st_onset']}, MMDAgg onset {d['agg_onset']}; loss log {len(d['loss_ep'])} epochs")

w0 = r50['vrt_onset']
w1 = r50['mmd_onset'] if SHADE_TO == 'MMD' else r50['c2st_onset']

# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.8), gridspec_kw={'hspace': 0.55, 'wspace': 0.42})

# (a) v7 loss
ax = axes[0, 0]
ax.plot(v7['loss_ep'], v7['loss_tr'], color=C_TRAIN, label='Train')
ax.plot(v7['loss_ep'], v7['loss_te'], color=C_TEST, ls='--', label='Test')
onset_line(ax, v7['vrt_onset'], C_VRT)
ax.text(v7['vrt_onset'] + 3, 1.72, 'VRT onset', fontsize=5, color=C_VRT, va='center')
ax.set_xlim(0, 199)
ax.set_ylim(0, 2.0)
ax.set_ylabel('Cross-entropy loss')
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.07), frameon=False, handlelength=1.6)
panel(ax, 'a', r'$\mathbf{v7}$  ResNet20, $D$=64')

# (b) v7 VRT
ax = axes[0, 1]
ax.plot(v7['vrt_ep'], neglog10(v7['vrt_p']), color=C_VRT)
alpha_line(ax, 192)
ax.text(192, 0.82, 'MMD, MMDAgg,\nED: null', fontsize=5.5, color=C_NULL, ha='right', va='center',
        multialignment='center', fontweight='bold', fontstyle='italic')
ax.set_xlim(0, 199)
ax.set_ylim(0, 3.8)
ax.set_ylabel(r'$-\log_{10}(p)$')
panel(ax, 'b', 'VRT')

# (c) v7 C2ST
ax = axes[0, 2]
ax.plot(v7['base_ep'], v7['c2st_acc'], color=C_C2ST)
ax.axhline(0.5, color=C_ALPHA, ls='--', lw=0.5, zorder=0)
ax.set_xlim(0, 199)
ax.set_ylim(0.48, 0.52)
ax.set_yticks([0.48, 0.49, 0.50, 0.51, 0.52])
ax.set_ylabel('C2ST accuracy')
panel(ax, 'c', 'C2ST')

# (d) r50 loss; the onset labels sit on the outer sides of the two lines, which are 12 epochs apart
ax = axes[1, 0]
ax.plot(r50['loss_ep'], r50['loss_tr'], color=C_TRAIN, label='Train')
ax.plot(r50['loss_ep'], r50['loss_te'], color=C_TEST, ls='--', label='Test')
onset_line(ax, r50['vrt_onset'], C_VRT)
onset_line(ax, r50['mmd_onset'], C_MMD)
ax.text(r50['vrt_onset'] - 2, 2.35, 'VRT\nonset', fontsize=5, color=C_VRT, ha='right', va='bottom')
ax.text(r50['mmd_onset'] + 2, 2.35, 'MMD\nonset', fontsize=5, color=C_MMD, ha='left', va='bottom')
ax.set_xlim(0, 200)
ax.set_ylim(0, 4.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Cross-entropy loss')
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.07), frameon=False, handlelength=1.6)
panel(ax, 'd', r'$\mathbf{r50}$  ResNet-50, $D$=2048')

# (e) r50 VRT + MMD
ax = axes[1, 1]
ax.axvspan(w0, w1, alpha=0.08, color=C_VRT, lw=0, zorder=0)
lr_drops(ax, [60, 120])
ax.plot(r50['vrt_ep'], neglog10(r50['vrt_p']), color=C_VRT, label='VRT', zorder=3)
ax.plot(r50['base_ep'], neglog10(r50['mmd_p']), color=C_MMD, ls=':', label='MMD', zorder=4)
alpha_line(ax, 193)
# Window marker above the level of 3.3 that both curves reach: a bracket over the shaded window, the label to its left
# (centered above it when the window is wide)
y_br = 3.58
ax.annotate('', xy=(w1, y_br), xytext=(w0, y_br),
            arrowprops=dict(arrowstyle='|-|, widthA=0.2, widthB=0.2', color=C_TXT, lw=0.7, shrinkA=0, shrinkB=0))
if w1 - w0 < 30:
    ax.text(w0 - 3, y_br, f'{w1 - w0} epochs', fontsize=6, fontweight='bold', color=C_TXT, ha='right', va='center')
else:
    ax.text((w0 + w1) / 2, y_br + 0.08, f'{w1 - w0} epochs', fontsize=6, fontweight='bold', color=C_TXT,
            ha='center', va='bottom')
ax.set_xlim(0, 200)
ax.set_ylim(0, 3.8)
ax.set_xlabel('Epoch')
ax.set_ylabel(r'$-\log_{10}(p)$')
ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), frameon=False, handlelength=1.6)
panel(ax, 'e', 'VRT + MMD')

# (f) r50 C2ST
ax = axes[1, 2]
lr_drops(ax, [60, 120])
ax.plot(r50['base_ep'], r50['c2st_acc'], color=C_C2ST)
ax.axhline(0.5, color=C_ALPHA, ls='--', lw=0.5, zorder=0)
ax.set_xlim(0, 200)
ax.set_ylim(0.48, 0.64)
ax.set_yticks([0.50, 0.55, 0.60])
ax.set_xlabel('Epoch')
ax.set_ylabel('C2ST accuracy')
panel(ax, 'f', 'C2ST')

for ax in axes[0]:
    ax.set_xticks([0, 50, 100, 150])
for ax in axes[1]:
    ax.set_xticks([0, 50, 100, 150, 200])

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
fig.savefig(os.path.join(OUT, 'fig2_trajectory.pdf'))
fig.savefig(os.path.join(OUT, 'fig2_trajectory.png'))
print(f'Saved to {OUT} (shaded window {w0} to {w1})')
plt.show()
