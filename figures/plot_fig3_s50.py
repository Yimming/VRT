"""
plot_fig3_s50.py — Fig 3: Regularization intervention (camera-ready).
VRT curves are medians over the 50 seeds (42 to 91) at 10,000 points per split, read from fig_data_s50/vrt_abl*.csv
(written by export_fig_data.py from the s50 results); the submitted figure used ablation_multiseed.log (10 seeds).
Onsets are computed from the curves with the paper's rule (median p < 0.05 at every later epoch): A 33, C 35.
Put this file next to the folder fig_data_s50.
"""
import csv, math, os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.join(ROOT, 'fig_data_s50')
OUT = ROOT

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.labelsize': 8,
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

C_A = '#2166AC'   # blue — baseline (weakest reg)
C_B = '#92C5DE'   # light blue — wd only
C_C = '#F4A582'   # light orange — aug only
C_D = '#D6191B'   # bright red — wd + aug (strongest reg)
C_ALPHA = '#BDBDBD'


def load_vrt(tag):
    """epochs and median p over the 50 seeds (fig_data_s50/vrt_<tag>.csv)"""
    with open(os.path.join(NEW, f'vrt_{tag}.csv')) as f:
        rows = list(csv.DictReader(f))
    return np.array([int(r['epoch']) for r in rows]), np.array([float(r['vrt_p']) for r in rows])


def neglog10(p_arr, floor=5e-4):
    return np.array([-math.log10(max(p, floor)) for p in p_arr])


def find_onset(p_arr, alpha=0.05):
    sig = p_arr < alpha
    for i in range(len(sig)):
        if np.all(sig[i:]):
            return i
    return None


# ── Load data (50-seed medians) ──
models = {name: load_vrt(tag) for name, tag in [('A_baseline', 'ablA'), ('B_strong_wd', 'ablB'),
                                                ('C_strong_aug', 'ablC'), ('D_wd_plus_aug', 'ablD')]}

print("Loaded models:")
for name, (eps, meds) in models.items():
    n_sig = np.sum(meds < 0.05)
    onset_idx = find_onset(meds)
    onset_str = f"ep {eps[onset_idx]}" if onset_idx is not None else "null"
    print(f"  {name}: {len(eps)} epochs, {n_sig}/{len(eps)} sig, onset={onset_str}")

# ── Figure ──
fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.2))

plot_cfg = {
    'A_baseline':     ('A: baseline',        C_A, '--', 0.9),
    'B_strong_wd':    ('B: 10$\\times$ wd',  C_B, '-',  0.9),
    'C_strong_aug':   ('C: strong aug',       C_C, '--', 0.9),
    'D_wd_plus_aug':  ('D: wd + aug',        C_D, '-',  0.9),
}

onset_info = []

for name, (eps, meds) in models.items():
    if name not in plot_cfg:
        continue
    label, color, ls, lw = plot_cfg[name]
    nlp = neglog10(meds)
    ax.plot(eps, nlp, color=color, ls=ls, lw=lw, label=label)

    onset_idx = find_onset(meds)
    if onset_idx is not None:
        onset_ep = eps[onset_idx]
        ax.vlines(onset_ep, 0, 3.0, color=color, ls='--', lw=0.6, alpha=0.5, zorder=0)
        onset_info.append((onset_ep, color, label[0]))

# Onset labels at consistent height; when two onsets are close (A 33, C 35), the earlier label goes to the left of its
# line so the two labels do not overlap
info = sorted(onset_info, key=lambda x: x[0])
for i, (onset_ep, color, letter) in enumerate(info):
    left = i + 1 < len(info) and info[i + 1][0] - onset_ep < 5
    ax.text(onset_ep - 0.5 if left else onset_ep + 0.5, 3.15, f'ep {onset_ep}',
            fontsize=4.5, color=color, va='bottom', ha='right' if left else 'left')

# Alpha line
y_alpha = -math.log10(0.05)
ax.axhline(y_alpha, color=C_ALPHA, ls='--', lw=0.5, zorder=0)
ax.text(48, y_alpha + 0.08, r'$\alpha\!=\!0.05$',
        fontsize=5, color='#999999', ha='right', va='bottom')

ax.set_xlim(0, 49)
ax.set_ylim(0, 3.8)
ax.set_xlabel('Epoch')
ax.set_ylabel(r'$-\log_{10}(p)$')
ax.set_title('VRT under regularization interventions', fontsize=7.5, loc='left')
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.08),
          frameon=False, handlelength=1.2, fontsize=5.5, ncol=2)

# ── Save ──
fig.savefig(os.path.join(OUT, 'fig3_regularization.pdf'))
fig.savefig(os.path.join(OUT, 'fig3_regularization.png'))
print(f"\nSaved to {OUT}")
plt.show()
