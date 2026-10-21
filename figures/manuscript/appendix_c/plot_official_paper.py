"""Final official-split plot from saved, exact aggregate rejection counts."""
from pathlib import Path
import os
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get('VRT_FIGURES_OUT', str(ROOT.parents[2] / 'results' / 'figures'))) / 'appendix_c'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 7,
    'axes.linewidth': .5,
    'xtick.major.width': .4,
    'xtick.major.size': 2.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'pdf.fonttype': 42,
    'savefig.dpi': 300,
})
methods = ['VRT', 'MMD', 'MMDAgg', 'C2ST', 'ED']
settings = ['ResNet-18\nCIFAR-10', 'ResNet-18\nCIFAR-100',
            'ResNet-50\nCIFAR-100', 'SimCLR\nCIFAR-10',
            'ResNet20-FRN\nCIFAR-10']
with (ROOT/'official_counts.csv').open(newline='') as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 25 and all(int(r['repetitions']) == 500 for r in rows)
values = [[100 * int(r['rejections']) / int(r['repetitions']) for r in rows[i*5:(i+1)*5]] for i in range(5)]
assert [r['method'] for r in rows] == methods * 5

fig, axes = plt.subplots(1, 5, figsize=(5.5, 1.68), sharey=True)
fig.subplots_adjust(left=.118, right=.996, bottom=.24, top=.77, wspace=.17)
for i, (ax, title, rates) in enumerate(zip(axes, settings, values)):
    ax.set_title(title, pad=6, linespacing=1.15)
    ax.set_xlim(-.8, 43)
    ax.set_ylim(4.5, -.5)
    ax.set_xticks([0, 20, 40])
    ax.set_yticks(range(5), methods)
    ax.tick_params(axis='y', length=0, pad=5)
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#E5E5E5', linewidth=.35)
    ax.spines['bottom'].set_color('#888888')
    ax.tick_params(axis='x', color='#888888')
    for y, value in enumerate(rates):
        color = '#2166AC' if y == 0 else '#555555'
        ax.hlines(y, 0, value, color=color, lw=.55, alpha=.65)
        ax.plot(value, y, 'o', ms=2.9, color=color)
        ax.annotate(f'{value:.1f}', (value, y), xytext=(2.5, 0),
                    textcoords='offset points', va='center', ha='left',
                    fontsize=5.8, color=color)
fig.supxlabel('Single-test rejection rate (%)', fontsize=7, y=.05)
fig.savefig(OUT / 'fig_cifar_official_control.pdf')
fig.savefig(OUT / 'fig_cifar_official_control.png')
plt.close(fig)
print(OUT / 'fig_cifar_official_control.pdf')
