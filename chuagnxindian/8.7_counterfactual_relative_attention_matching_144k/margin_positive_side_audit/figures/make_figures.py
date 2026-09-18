#!/usr/bin/env python3
"""Reproducible data figures for the 8.7 margin/positive-side audit."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
BLUE='#0072B2'; ORANGE='#D55E00'; GRAY='#777777'
def save(fig,name):
    fig.tight_layout();fig.savefig(HERE/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(HERE/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def main():
    margin=pd.read_csv(ROOT/'margin/per_sample_margin.csv'); is_a=margin.group.eq('A_ogl_help')
    fig,ax=plt.subplots(figsize=(5,3));bp=ax.boxplot([margin.loc[~is_a,'margin']*1e5,margin.loc[is_a,'margin']*1e5],tick_labels=['non-Group-A\n(n=4,730)','fixed Group-A\n(n=428)'],showfliers=False,patch_artist=True)
    for box,color in zip(bp['boxes'],[BLUE,ORANGE]):box.set_facecolor(color);box.set_alpha(.75)
    ax.axhline(0,color='black',lw=.8,ls='--');ax.set_ylabel('Margin Dneg_mean - Dpos (x1e-5)');ax.set_title('No Group-A margin/violation enrichment');save(fig,'margin_groupA_distribution')
    fig,ax=plt.subplots(figsize=(5,3));ax.scatter(margin.loc[~is_a,'margin']*1e5,margin.loc[~is_a,'delta_iou'],s=4,color=BLUE,alpha=.2,label='non-Group-A');ax.scatter(margin.loc[is_a,'margin']*1e5,margin.loc[is_a,'delta_iou'],s=8,color=ORANGE,alpha=.55,label='fixed Group-A');ax.axvline(0,color='black',lw=.8,ls='--');ax.set_xlabel('Margin (x1e-5)');ax.set_ylabel('OGL - AUD IoU');ax.set_title('Spearman rho=0.022, p=.114');ax.legend(frameon=False,markerscale=2);save(fig,'margin_vs_ogl_gain')
    four=pd.read_csv(ROOT/'satisfied_but_over/four_way_partition.csv');labels=['sat.\nnot OVER','sat.\nOVER','viol.\nnot OVER','viol.\nOVER'];fig,ax=plt.subplots(figsize=(5,3));bars=ax.bar(labels,four.groupA_fraction,color=[BLUE,ORANGE,GRAY,ORANGE]);ax.set_ylim(0,.14);ax.set_ylabel('Fixed Group-A fraction');ax.set_title('Residual OVER persists despite satisfied CRAM')
    for bar,value in zip(bars,four.groupA_fraction):ax.text(bar.get_x()+bar.get_width()/2,value+.004,f'{value:.3f}',ha='center',fontsize=8)
    save(fig,'satisfied_but_overactivated')
    positive=pd.read_csv(ROOT/'positive_map/groupA_positive_summary.csv');wanted=['precision','coverage','activated_area','nearfp_response'];positive=positive[positive.metric.isin(wanted)].set_index('metric');fig,axes=plt.subplots(1,4,figsize=(10,2.5));titles=['Precision','Coverage','Activated area','NearFP response']
    for ax,key,title in zip(axes,wanted,titles):
        row=positive.loc[key];values=[row.right_mean,row.left_mean];ax.bar(['non-A','Group-A'],values,color=[BLUE,ORANGE]);ax.set_title(title,fontsize=9);ax.text(.5,max(values)*1.02,f'delta={row.mean_diff_left_minus_right:+.3f}',ha='center',fontsize=8)
    fig.suptitle('Matched AUD positive-map signature (VGG-SS 144k)',y=1.04);save(fig,'groupA_positive_map_signature')
if __name__=='__main__':main()
