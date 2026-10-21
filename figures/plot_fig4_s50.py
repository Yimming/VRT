"""
plot_fig4_s50.py — Fig 4: Mechanistic analysis (1x2), camera-ready
(a) Synthetic power profiles at matched budget (both tests on the same 5,000 points per sample, 2,000 permutations,
    VRT k = 60, M = 2,000, MMD Gaussian kernel with the median heuristic; D = 500, 200 repetitions). Two class-cluster
    alternatives in which the training sample changes inside its classes (within-class concentration + 5 percent,
    concentrated subpopulation of 10 percent at 3 kappa_w; fig_data_s50/fig4a_clusters.csv, synth_fig4a_clusters.py,
    C = 10 classes), and a single-cloud vMF mean-direction shift of 2 degrees (fig_data_s50/fig4a_matched.csv
    at budget 5,000, synth_fig4a_matched.py). Both tests take l2-normalized input. The remaining conditions,
    including single-cloud concentration and radial spread, are retained in Appendix D.
(b) ResNet-50 (CIFAR-100): sustained onset (paper rule, median p over the 50 seeds 42 to 91 at 10,000 points per split)
    under eval-mode and train-mode BatchNorm extraction for MMD and ED on unnormalized (raw) and l2-normalized features and
    for VRT, drawn as filled (eval) and open (train) markers joined by an arrow; from fig_data_s50/fig4b_r50.csv
    (export_fig_data.py). Onsets: MMD raw 63 -> 121, MMD normalized 73 -> 121, ED raw 62 -> 121, ED normalized 73 -> 121,
    VRT 61 in both modes. Normalized rows are drawn in a lighter red because their onsets are also in Table 2.
The submitted panel (b) (ImageNet MMD, eval versus train mode, six checkpoints) is superseded by Table 2.
Put this file next to the folder fig_data_s50.
"""
import csv
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.join(ROOT, 'fig_data_s50')
OUT = ROOT

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
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

C_VRT = '#2166AC'
C_MMD = '#D6191B'
C_ALPHA = '#BDBDBD'

fig = plt.figure(figsize=(5.5, 2.3))
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.35, 1.0], wspace=0.38, left=0.075, right=0.985, bottom=0.33, top=0.90)
ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

# ════════════════════════════════════════════
# (a) Synthetic power: two class-cluster alternatives and one single-cloud mean-direction shift,
#     all at matched budget (5,000 points per sample for both tests,
#     2,000 permutations; synth_fig4a_clusters.py and synth_fig4a_matched.py)
# ════════════════════════════════════════════
labels_a = [
    'Within-class\nconcentration\n(+5%)',
    'Concentrated\nsubpopulation\n(10%)',
    'Mean shift\n(2.0°)',
]
GROUPS = [  # (csv kind, condition name)
    ('clusters', 'tight_0.05'),
    ('clusters', 'memo_0.10'),
    ('matched', 'mean_2deg'),
]
MATCHED_BUDGET = 5000
CLUSTER_C = 10


def rates(path, keep):
    """keep(row) -> True for the rows of the wanted budget / class count; returns {(cond, test): (rejections, n)}"""
    cnt, tot = {}, {}
    for r in csv.DictReader(open(path)):
        if not keep(r):
            continue
        k = (r['cond'], r['test']); tot[k] = tot.get(k, 0) + 1; cnt[k] = cnt.get(k, 0) + (float(r['p']) < 0.05)
    return {k: (cnt[k], tot[k]) for k in tot}


matched_csv = os.path.join(NEW, 'fig4a_matched.csv')
clusters_csv = os.path.join(NEW, 'fig4a_clusters.csv')
R = {'matched': rates(matched_csv, lambda r: int(r['budget']) == MATCHED_BUDGET)}
R['clusters'] = rates(clusters_csv, lambda r: int(r['C']) == CLUSTER_C)
for kind, condition in GROUPS:
    for test in ('VRT', 'MMD'):
        assert R[kind][(condition, test)][1] == 200


def pct(kind, cond, test):
    v = R[kind].get((cond, test))
    return None if v is None else 100 * v[0] / v[1]


vrt_power = [pct(k, c, 'VRT') for k, c in GROUPS]
mmd_power = [pct(k, c, 'MMD') for k, c in GROUPS]
for (k, c), lab in zip(GROUPS, labels_a):
    v, m = R[k].get((c, 'VRT')), R[k].get((c, 'MMD'))
    print(f"{lab.replace(chr(10), ' '):42s} VRT {v[0] if v else '-'}/{v[1] if v else '-'}   MMD {m[0] if m else '-'}/{m[1] if m else '-'}")

x_a = np.arange(len(labels_a))
w_a = 0.32 * 3.15 / 5.15  # preserve the original physical bar width
for vals, off, color, lab in ((vrt_power, -w_a / 2, C_VRT, 'VRT'), (mmd_power, w_a / 2, C_MMD, 'MMD')):
    xs = [x_a[i] + off for i, v in enumerate(vals) if v is not None]
    ys = [v for v in vals if v is not None]
    bars = ax_a.bar(xs, ys, w_a, color=color, label=lab, edgecolor='white', linewidth=0.4)
    for bar in bars:
        h = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f'{h:g}', ha='center', va='bottom', fontsize=5.5, color=color)

ax_a.axhline(5, color=C_ALPHA, ls='--', lw=0.5, zorder=0)
ax_a.text(2.58, 13.0, r'$\alpha\!=\!0.05$', fontsize=5, color='#999999', ha='right', va='bottom')   # above the bar labels
for xv in (1.5,):
    ax_a.axvline(xv, color='#DDDDDD', lw=0.6, zorder=0)
ax_a.text(0.5, 117, '10 class clusters', ha='center', va='top', fontsize=5.5, color='#666666')
ax_a.text(2.0, 117, 'Single cluster', ha='center', va='top', fontsize=5.5, color='#666666')

ax_a.set_xticks(x_a)
ax_a.set_xticklabels(labels_a, fontsize=5.6)
ax_a.set_ylabel('Rejection rate (%)')
ax_a.set_xlim(-0.55, 2.6)
ax_a.set_ylim(0, 118)
ax_a.set_yticks([0, 25, 50, 75, 100])
ax_a.set_title('(a) Synthetic power profiles', fontsize=7.5, loc='left')
ax_a.legend(loc='upper left', bbox_to_anchor=(0.23, 0.82), frameon=False, handlelength=1.0, fontsize=5.5)

# ════════════════════════════════════════════
# (b) ResNet-50: sustained onset under eval-mode and train-mode extraction (onset shift chart)
# ════════════════════════════════════════════
rows = list(csv.DictReader(open(os.path.join(NEW, 'fig4b_r50.csv'))))
eps = np.array([int(r['epoch']) for r in rows])


def col(name):
    return np.array([float(r[name]) for r in rows])


def onset(p, alpha=0.05):
    sig = p < alpha
    return next((int(eps[i]) for i in range(len(sig)) if sig[i:].all()), None)


C_MMD_L = '#F4A6A0'   # normalized features, de-emphasized (their eval -> train onsets are also in Table 2)
items = [  # bottom to top: (label, eval column, train column, color)
    ('ED normalized', 'ed_norm_eval', 'ed_norm_train', C_MMD_L),
    ('ED raw', 'ed_raw_eval', 'ed_raw_train', C_MMD),
    ('MMD normalized', 'mmd_norm_eval', 'mmd_norm_train', C_MMD_L),
    ('MMD raw', 'mmd_raw_eval', 'mmd_raw_train', C_MMD),
    ('VRT', 'vrt_eval', 'vrt_train', C_VRT),
]
for i, (label, ce, ct, color) in enumerate(items):
    oe, ot = onset(col(ce)), onset(col(ct))
    print(f'{label:15s} eval {oe}  train {ot}')
    if oe != ot:
        ax_b.annotate('', xy=(ot, i), xytext=(oe, i),
                      arrowprops=dict(arrowstyle='-|>', color=color, lw=1.0, shrinkA=3, shrinkB=2, mutation_scale=6))
        ax_b.plot(oe, i, 'o', color=color, ms=4, zorder=3)
        ax_b.plot(ot, i, 'o', mfc='white', mec=color, ms=4, mew=1.0, zorder=4)
        ax_b.text(oe - 3, i, str(oe), ha='right', va='center', fontsize=5, color=color)
        ax_b.text(ot + 3, i, str(ot), ha='left', va='center', fontsize=5, color=color)
    else:                                   # same onset in both modes: the two markers stacked at one epoch
        ax_b.plot(oe, i + 0.13, 'o', color=color, ms=4, zorder=3)
        ax_b.plot(ot, i - 0.13, 'o', mfc='white', mec=color, ms=4, mew=1.0, zorder=4)
        ax_b.text(oe + 4, i, str(oe), ha='left', va='center', fontsize=5, color=color)

ax_b.set_yticks(range(len(items)))
ax_b.set_yticklabels([it[0] for it in items], fontsize=6)
ax_b.set_xlim(40, 140)
ax_b.set_ylim(-0.6, len(items) - 0.4)
ax_b.set_xticks([40, 60, 80, 100, 120, 140])
ax_b.set_xlabel('Sustained onset (epoch)')
ax_b.plot([], [], 'o', color='#555555', ms=4, label='eval mode (filled)')
ax_b.plot([], [], 'o', mfc='white', mec='#555555', ms=4, mew=1.0, label='train mode (open)')
ax_b.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=5, handletextpad=0.3, labelspacing=0.3)
ax_b.set_title('(b) ResNet-50, eval vs train mode', fontsize=7.5, loc='left')

fig.savefig(os.path.join(OUT, 'fig4_diagnosis.pdf'))
fig.savefig(os.path.join(OUT, 'fig4_diagnosis.png'))
print(f"Saved to {OUT}")
plt.close(fig)
