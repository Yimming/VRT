"""One figure of shared-axis onset bands, using the audited 24-model snapshot.

Each displayed model group has VRT, MMD on c-hat, and embedding MMD in a
fixed top-to-bottom order. Zoo ranges use paired endpoints and no midpoint.
No delta annotations. All omitted model records are retained in the JSON.
"""
from pathlib import Path
import argparse, json, os

ROOT=Path(__file__).resolve().parent
OUT = Path(os.environ.get('VRT_FIGURES_OUT', str(ROOT.parents[2] / 'results' / 'figures'))) / 'appendix_i'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

METHODS=[('vrt','VRT','#2166AC','o'),
         ('chat_local',r'MMD on $\hat c$','#7B3294','D'),
         ('embedding','MMD on emb.','#D55E00','s')]
TOP=[('ResNet20-FRN',['v7']),('ResNet-50',['r50']),
     ('Zoo CIFAR-10 (5)',[f'zoo_c10_m{i}' for i in range(5)]),
     ('Zoo CIFAR-100 (5)',[f'zoo_c100_m{i}' for i in range(5)]),
     ('Configuration A',['ablA']),('Configuration C',['ablC'])]
BOTTOM=[('SimCLR CIFAR-10',['simclr_c10']),('SimCLR CIFAR-100',['simclr_c100'])]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=ROOT/'onsets.json')
    a=ap.parse_args()
    records=json.loads(a.data.read_text());data={r['tag']:r for r in records}
    assert len(data)==24 and all(r['claimed_value_verified'] for r in records)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],'font.size':9,
        'mathtext.fontset':'stix','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
        'axes.linewidth':.5,'xtick.major.width':.45,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(5.5,5.0))
    handles=[Line2D([0],[0],color=c,marker=m,lw=1.2,markersize=4.3,label=l)
             for _,l,c,m in METHODS]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.60,.999),
               ncols=3,frameon=False,fontsize=9.2,handlelength=1.4,
               handletextpad=.45,columnspacing=1.2,borderaxespad=0)
    blocks=[(TOP,80,[.278,.375,.695,.553]),
            (BOTTOM,320,[.278,.097,.695,.198])]
    plotted=[];used=set();stride=3.6
    for bi,(rows,xmax,rect) in enumerate(blocks):
        ax=fig.add_axes(rect)
        ax.set_xlim(0,xmax)
        ax.set_ylim((len(rows)-1)*stride+2.6,-.6)
        centers=[i*stride+1 for i in range(len(rows))]
        ax.set_yticks(centers,[label for label,_ in rows])
        ax.tick_params(axis='y',length=0,pad=9,labelsize=9)
        ax.set_xticks([0,20,40,60,80] if bi==0 else [0,80,160,240,320])
        ax.tick_params(axis='x',labelsize=8.5,pad=2,length=2.6)
        ax.set_xlabel('Epoch',fontsize=8.8,labelpad=2)
        # Quiet separators span labels and marks, centered between model groups.
        trans=blended_transform_factory(fig.transFigure,ax.transData)
        for i in range(len(rows)-1):
            y=i*stride+(2+stride)/2
            ax.plot([.035,.973],[y,y],transform=trans,color='#D8D8D8',
                    lw=.35,clip_on=False,zorder=0)
        for ri,(label,tags) in enumerate(rows):
            used.update(tags)
            for mi,(field,method,color,marker) in enumerate(METHODS):
                y=ri*stride+mi
                vals=[data[tag][field] for tag in tags]
                finite=[v for v in vals if v is not None]
                assert len(finite) in [0,len(tags)]
                p=dict(block=bi,model_group=label,tags=tags,method=field,
                       exact_values=vals,n_detected=len(finite))
                if not finite:
                    ax.text(.975,y,'ND',transform=ax.get_yaxis_transform(),
                            fontsize=8.8,ha='right',va='center',color='black')
                    p.update(minimum=None,maximum=None,displayed='ND')
                else:
                    low,high=min(finite),max(finite)
                    # Length encodes onset. The zoo's heavier endpoint segment
                    # shows its observed range, not a synthetic central value.
                    ax.plot([0,low],[y,y],color=color,lw=1.2,
                            solid_capstyle='butt',zorder=2)
                    if len(tags)==1:
                        ax.scatter([low],[y],s=22 if marker=='D' else 19,
                                   color=color,marker=marker,edgecolors='white',
                                   linewidths=.3,zorder=3)
                        text=str(low)
                    else:
                        ax.plot([low,high],[y,y],color=color,lw=3,
                                solid_capstyle='butt',zorder=3)
                        for x in [low,high]:
                            ax.plot([x,x],[y-.18,y+.18],color=color,lw=.9,zorder=3)
                        text=f'{low}-{high}' if low!=high else str(low)
                    ax.annotate(text,(high,y),xytext=(4,0),textcoords='offset points',
                                fontsize=8.8,ha='left',va='center',color='black',zorder=4)
                    p.update(minimum=low,maximum=high,displayed=text)
                plotted.append(p)
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/'fig_mmd_on_chat.pdf')
    fig.savefig(OUT/'fig_mmd_on_chat.png',dpi=220)
    plt.close(fig)
    omitted=[r for r in records if r['tag'] not in used]
    assert len(plotted)==24 and len(used)==16 and len(omitted)==8
    both=[r for r in records if r['vrt'] is not None and r['chat_local'] is not None]
    paired=dict(both_detect=len(both),chat_earlier=sum(r['chat_local']<r['vrt'] for r in both),
                equal=sum(r['chat_local']==r['vrt'] for r in both),
                chat_later=sum(r['chat_local']>r['vrt'] for r in both),
                max_abs_gap=max(abs(r['chat_local']-r['vrt']) for r in both))
    assert paired==dict(both_detect=19,chat_earlier=9,equal=9,chat_later=1,max_abs_gap=14)
    (OUT/'aggregation.json').write_text(json.dumps(dict(source=a.data.name,
       size_inches=[5.5,5.0],minimum_font_pt=8.5,
       methods=[dict(field=f,label=l,color=c,marker=m) for f,l,c,m in METHODS],
       plotted=plotted,omitted_exact_records=omitted,all24_exact_records=records,
       paired_counts_before_aggregation=paired),indent=2)+'\n')
    print('Created one shared-axis horizontal figure with 8 model groups and no delta annotations.')


if __name__=='__main__':main()
