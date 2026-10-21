"""
plot_trajectories_s50.py — Appendix figure (Figure 6): per-model loss + VRT trajectories (camera-ready)
=======================================================================================================
Zoo C10, Zoo C100, Wild Park, Configurations A/B/C/D.
VRT curves: medians over the 50 seeds (42 to 91) at 10,000 points per split, fig_data_s50/vrt_<tag>.csv (written by
export_fig_data.py). The submitted figure plotted the single-run benchmark_*.csv p-values. Onsets are read from
fig_data_s50/onsets.csv (paper's rule): Zoo C10 m0 26, Zoo C100 m0 34, A 33, C 35; Wild Park, B and D none.
Loss curves are unchanged: benchmark_final/losses_*.csv.
Put this file next to the folders fig_data_s50 and benchmark_final.
"""

import csv
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(ROOT, 'benchmark_final')        # loss curves (unchanged)
NEW = os.path.join(ROOT, 'fig_data_s50')           # 50-seed VRT medians and onsets
OUT_ROOT = ROOT
OUT_FIG = os.path.join(OUT_ROOT, 'Figures')
os.makedirs(OUT_FIG, exist_ok=True)


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',

    'font.size': 7.2,
    'axes.labelsize': 6.4,
    'axes.titlesize': 6.8,
    'xtick.labelsize': 5.8,
    'ytick.labelsize': 5.8,
    'legend.fontsize': 5.2,

    'axes.linewidth': 0.5,
    'xtick.major.width': 0.4,
    'ytick.major.width': 0.4,
    'xtick.major.size': 2.2,
    'ytick.major.size': 2.2,
    'xtick.major.pad': 1.5,
    'ytick.major.pad': 1.5,

    'lines.linewidth': 0.9,
    'axes.spines.top': False,
    'axes.spines.right': False,

    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

# Keep fonts embedded as TrueType in PDF.
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


# ---------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------
C_TRAIN = '#2166AC'
C_TEST = '#B2182B'
C_VRT = '#1B7837'
C_ALPHA = '#BDBDBD'
C_NULL = '#999999'


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_csv(path):
    if not os.path.exists(path):
        print(f'[WARN] Missing file: {path}')
        return None
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def col(rows, key, dtype=float):
    out = []
    for r in rows:
        try:
            out.append(dtype(r[key]))
        except Exception:
            out.append(float('nan'))
    return np.array(out)


def neglog10(p_arr, floor=5e-4):
    vals = []
    for p in p_arr:
        if np.isnan(p):
            vals.append(0.0)
        else:
            vals.append(-math.log10(max(float(p), floor)))
    return np.array(vals)


def get_loss_keys(data):
    """Find the correct column names for train/test loss."""
    keys = data[0].keys()
    candidates = [
        ('tr_loss', 'te_loss'),
        ('train_loss', 'test_loss'),
        ('loss_tr', 'loss_te'),
    ]
    for tr, te in candidates:
        if tr in keys and te in keys:
            return tr, te
    return None, None


def add_onset_label(ax, onset, x_values):
    """
    Add a small onset label that avoids overlapping curves too much.
    x is in data coordinates, y is in axes coordinates.
    """
    if onset is None or len(x_values) == 0:
        return

    x0, x1 = x_values[0], x_values[-1]
    xrange = x1 - x0 if x1 != x0 else 1.0

    # Put text to the right unless onset is near the right edge.
    if onset < x0 + 0.75 * xrange:
        x_text = onset + 0.025 * xrange
        ha = 'left'
    else:
        x_text = onset - 0.025 * xrange
        ha = 'right'

    ax.text(
        x_text,
        0.88,
        f'ep {onset}',
        transform=ax.get_xaxis_transform(),
        fontsize=4.7,
        color=C_VRT,
        ha=ha,
        va='center',
        bbox=dict(
            facecolor='white',
            edgecolor='none',
            alpha=0.80,
            pad=0.45,
        ),
        clip_on=False,
        zorder=10,
    )


def load_onsets():
    rows = load_csv(os.path.join(NEW, 'onsets.csv'))
    return {r['tag']: (int(r['vrt_onset']) if r['vrt_onset'] else None) for r in rows}


ONSET = load_onsets()


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
def model(name, loss_file, tag):
    return {'name': name, 'loss_path': os.path.join(OLD, loss_file),
            'vrt_path': os.path.join(NEW, f'vrt_{tag}.csv'), 'onset': ONSET[tag]}


models = [
    model('Zoo C10 ($D{=}512$)', 'losses_zoo_c10_m0.csv', 'zoo_c10_m0'),
    model('Zoo C100 ($D{=}512$)', 'losses_zoo_c100_m0.csv', 'zoo_c100_m0'),
    model('Wild Park CNN ($D{=}32$)', 'losses_wp.csv', 'wp'),
    model('Config A: baseline ($D{=}512$)', 'losses_ablation_A_baseline.csv', 'ablA'),
    model('Config B: $10{\\times}$ wd ($D{=}512$)', 'losses_ablation_B_strong_wd.csv', 'ablB'),
    model('Config C: strong aug ($D{=}512$)', 'losses_ablation_C_strong_aug.csv', 'ablC'),
    model('Config D: wd + aug ($D{=}512$)', 'losses_ablation_D_wd_plus_aug.csv', 'ablD'),
]
for m in models:
    print(f"{m['name']:40s} onset {m['onset']}")


# ---------------------------------------------------------------------
# Plot one model
# ---------------------------------------------------------------------
def plot_model(ax_l, ax_r, m, label, show_xlabel=False):
    # -------------------------
    # Loss panel
    # -------------------------
    data = load_csv(m['loss_path'])
    if data:
        loss_ep = col(data, 'epoch', int)
        tr_key, te_key = get_loss_keys(data)

        if tr_key is not None:
            loss_tr = col(data, tr_key)
            loss_te = col(data, te_key)

            ax_l.plot(loss_ep, loss_tr, color=C_TRAIN, lw=0.9)
            ax_l.plot(loss_ep, loss_te, color=C_TEST, lw=0.9, ls='--')

            valid = loss_te[~np.isnan(loss_te)]
            if len(valid) > 2:
                med = np.median(valid)
                ymax = min(np.max(valid) * 1.1, med * 4)
                if np.isfinite(ymax) and ymax > 0:
                    ax_l.set_ylim(0, ymax)
        else:
            ax_l.text(
                0.5, 0.5, 'Loss columns not found',
                transform=ax_l.transAxes,
                ha='center', va='center',
                fontsize=6.2, color='gray'
            )

        if len(loss_ep) > 0:
            ax_l.set_xlim(loss_ep[0], loss_ep[-1])

        if m['onset'] is not None:
            ax_l.axvline(m['onset'], color=C_VRT, ls='--', lw=0.6, alpha=0.55)
    else:
        ax_l.text(
            0.5, 0.5, 'Loss data not found',
            transform=ax_l.transAxes,
            ha='center', va='center',
            fontsize=6.2, color='gray'
        )

    ax_l.set_ylabel('CE loss')
    if show_xlabel:
        ax_l.set_xlabel('Epoch')
    else:
        ax_l.set_xlabel('')
        ax_l.tick_params(labelbottom=False)

    ax_l.set_title(m['name'], fontsize=6.8, loc='left', pad=1.2)
    ax_l.text(
        -0.13, 1.04, label,
        transform=ax_l.transAxes,
        fontsize=8.0,
        fontweight='bold',
        va='bottom'
    )

    # -------------------------
    # VRT panel
    # -------------------------
    data = load_csv(m['vrt_path'])
    if data:
        vrt_ep = col(data, 'epoch', int)
        vrt_p = col(data, 'vrt_p')
        nlp = neglog10(vrt_p)

        ax_r.plot(vrt_ep, nlp, color=C_VRT, lw=0.9)
        ax_r.axhline(-math.log10(0.05), color=C_ALPHA, ls='--', lw=0.5)
        ax_r.set_ylim(0, 3.8)

        if len(vrt_ep) > 0:
            ax_r.set_xlim(vrt_ep[0], vrt_ep[-1])

        if m['onset'] is not None:
            ax_r.axvline(m['onset'], color=C_VRT, ls='--', lw=0.6, alpha=0.55)
            add_onset_label(ax_r, m['onset'], vrt_ep)
        else:
            ax_r.text(
                0.5, 0.86, 'null',
                transform=ax_r.transAxes,
                fontsize=5.8,
                color=C_NULL,
                ha='center',
                va='center',
                fontstyle='italic',
                bbox=dict(
                    facecolor='white',
                    edgecolor='none',
                    alpha=0.65,
                    pad=0.35,
                ),
            )
    else:
        ax_r.text(
            0.5, 0.5, 'VRT data not found',
            transform=ax_r.transAxes,
            ha='center', va='center',
            fontsize=6.2, color='gray'
        )

    ax_r.set_ylabel(r'$-\log_{10}(p)$')
    if show_xlabel:
        ax_r.set_xlabel('Epoch')
    else:
        ax_r.set_xlabel('')
        ax_r.tick_params(labelbottom=False)


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
n_models = len(models)

fig, axes = plt.subplots(
    n_models, 2,
    figsize=(5.35, 9.85),
    gridspec_kw={
        'hspace': 0.78,
        'wspace': 0.34,
    }
)

labels = [f'({chr(97 + i)})' for i in range(n_models)]

for i, m in enumerate(models):
    show_xlabel = (i == n_models - 1)
    plot_model(axes[i, 0], axes[i, 1], m, labels[i], show_xlabel=show_xlabel)


# Shared legend, instead of per-panel legends.
legend_handles = [
    Line2D([0], [0], color=C_TRAIN, lw=0.9, label='Train'),
    Line2D([0], [0], color=C_TEST, lw=0.9, ls='--', label='Test'),
    Line2D([0], [0], color=C_VRT, lw=0.9, label='VRT'),
    Line2D([0], [0], color=C_ALPHA, lw=0.5, ls='--', label=r'$\alpha=0.05$'),
]

fig.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.992),
    ncol=4,
    frameon=False,
    handlelength=1.6,
    columnspacing=0.9,
    handletextpad=0.4,
)

fig.subplots_adjust(
    top=0.955,
    bottom=0.040,
    left=0.115,
    right=0.985,
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
pdf_path = os.path.join(OUT_FIG, 'fig_trajectories.pdf')
png_path = os.path.join(OUT_FIG, 'fig_trajectories.png')

fig.savefig(pdf_path)
fig.savefig(png_path)

fig.savefig(os.path.join(OUT_ROOT, 'fig_trajectories_appendix.pdf'))
fig.savefig(os.path.join(OUT_ROOT, 'fig_trajectories_appendix.png'))

print(f"Saved PDF: {pdf_path}")
print(f"Saved PNG: {png_path}")

plt.show()
