"""Appendix F, six overview panels and four regularization controls.

Uses validated loaders from the approved preview. Full unsmoothed loss
trajectories; SimCLR in eval mode, converted to NT-Xent nats. No shading.
"""
from pathlib import Path
import os
import sys
import json

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get('VRT_FIGURES_OUT', str(ROOT.parents[2] / 'results' / 'figures'))) / 'appendix_f'
OUT.mkdir(parents=True, exist_ok=True)
from plot_loss_with_simclr import (load, onsets, audit, plt, BLUE, RED, GREEN,
                                  load_ssl, ssl_audit, source_hashes, draw_panel)
from matplotlib.lines import Line2D


def main():
    width, height = 5.5, 7.2
    fig = plt.figure(figsize=(width, height))
    # Positions are in inches, retaining the approved overview panel sizes.
    overview = [
        ('v7','ResNet20-FRN','CIFAR-10',.44,5.5),
        ('r50','ResNet-50','CIFAR-100',2.2275,5.5),
        ('wp','WildPark','CIFAR-10',4.015,5.5),
        ('zoo_c100_m0','ResNet-18','CIFAR-100 Zoo',.44,3.7),
        ('cifar10','SimCLR','CIFAR-10',2.2275,3.7),
        ('cifar100','SimCLR','CIFAR-100',4.015,3.7),
    ]
    for i,(tag,model,dataset,x,y) in enumerate(overview):
        ssl = model == 'SimCLR'
        data,onset = load_ssl(tag,229 if tag=='cifar10' else 191) if ssl else (load(tag),onsets[tag])
        if ssl:
            data = data.copy()
            data[:, 1:3] *= 0.07 / 0.5
        ax=fig.add_axes([x/width,y/height,1.2925/width,1.10/height])
        draw_panel(ax,data,onset,model,dataset,chr(97+i),ssl=ssl,ylabel=i in (0,3))
        if ssl:
            ax.set_ylim(4.9, 7.0)
            ax.set_yticks([5, 6, 7])
            ax.set_ylabel('Contrastive loss (nats)', labelpad=4)
    controls = [
        ('ablA','A  Baseline',.55,1.90),
        ('ablB',r'B  $10\times$ weight decay',3.05,1.90),
        ('ablC','C  Strong augmentation',.55,.35),
        ('ablD','D  Weight decay + augmentation',3.05,.35),
    ]
    for i,(tag,title,x,y) in enumerate(controls):
        ax=fig.add_axes([x/width,y/height,2.05/width,1.00/height])
        draw_panel(ax,load(tag),onsets[tag],'ResNet-18','CIFAR-10',chr(103+i),ylabel=i%2==0)
        ax.set_xlim(0,50)
        ax.set_xticks([0,25,50])
        ax.set_title(f'({chr(103+i)}) {title}',loc='left',pad=6,fontsize=8)
    handles=[Line2D([],[],color=BLUE,lw=1,label='Train'),
             Line2D([],[],color=RED,lw=1,ls='--',label='Test'),
             Line2D([],[],color=GREEN,lw=.85,ls='--',label='VRT onset')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,1.0),
               ncol=3,frameon=False,fontsize=8,columnspacing=1.7)
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'fig_trajectories_appendix.{ext}')
    plt.close(fig)
    (OUT/'figure_audit.json').write_text(json.dumps({
        'supervised':audit,'simclr':ssl_audit,'source_sha256':source_hashes,
        'panel_order':[r[0] for r in overview]+[r[0] for r in controls],
        'size_inches':[width,height],
        'simclr_plot_scale':0.07/0.5,
        'simclr_plot_units':'NT-Xent in nats, original CSV losses multiplied by0.14',
        'zoo_selection':'CIFAR-100 m0, one model rather than an across-model average',
        'losses':'Unsmoothed checkpoint measurements; not medians over VRT seeds',
        'epoch0':'Omitted only from supervised loss display, as before'
    },indent=2))
    print({tag:record['onset'] for tag,record in audit.items()})
    print({tag:record['vrt_onset_recomputed'] for tag,record in ssl_audit.items()})

if __name__=='__main__': main()
