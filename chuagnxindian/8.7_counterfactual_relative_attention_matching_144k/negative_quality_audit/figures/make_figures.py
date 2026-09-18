#!/usr/bin/env python3
"""Publication-safe figures for the read-only negative-pool audit."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
OUT=HERE
COL={'groupA':'#D55E00','non':'#0072B2','help':'#CC79A7','other':'#999999'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def save(fig,name):
    fig.tight_layout(); fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight'); fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight'); plt.close(fig)
def main():
    x=pd.read_csv(ROOT/'groupA/vggss_144k_pool_with_fixed_residual_labels.csv')
    a=x.group.eq('A_ogl_help')
    # Fixed Group-A definition; lower ratio means top-4 candidates are more
    # different from the random draw.  All selection remained label-free.
    fig,ax=plt.subplots(figsize=(5.4,3.2))
    data=[x.loc[~a,'opportunity_gap_mean']*1e5,x.loc[a,'opportunity_gap_mean']*1e5]
    bp=ax.boxplot(data,labels=['non-Group-A\n(n=4,730)','fixed Group-A\n(n=428)'],showfliers=False,patch_artist=True)
    for p,c in zip(bp['boxes'],[COL['non'],COL['groupA']]): p.set_facecolor(c);p.set_alpha(.75)
    ax.set_ylabel(r'Opportunity gap $\times 10^5$');ax.set_title('VGG-SS 144k: large-pool opportunity')
    save(fig,'opportunity_gap_distribution')
    fig,ax=plt.subplots(figsize=(5.4,3.2))
    rates=[x.loc[~a,'missed_confusing_negative'].mean(),x.loc[a,'missed_confusing_negative'].mean()]
    ax.bar(['non-Group-A','fixed Group-A'],rates,color=[COL['non'],COL['groupA']]);ax.set_ylim(0,.4);ax.set_ylabel('Missed confusing-negative rate');ax.set_title('VGG-SS 144k: no Group-A enrichment')
    for i,v in enumerate(rates):ax.text(i,v+.012,f'{v:.3f}',ha='center')
    save(fig,'missed_negative_rate')
    fig,ax=plt.subplots(figsize=(5.4,3.2))
    ax.scatter(x.loc[~a,'opportunity_gap_mean']*1e5,x.loc[~a,'delta_iou'],s=5,alpha=.22,color=COL['non'],label='non-Group-A')
    ax.scatter(x.loc[a,'opportunity_gap_mean']*1e5,x.loc[a,'delta_iou'],s=8,alpha=.55,color=COL['groupA'],label='fixed Group-A')
    rho,p=spearmanr(x.opportunity_gap_mean,x.delta_iou)
    ax.text(.03,.95,f'Spearman $\\rho$={rho:.3f}, p={p:.3g}',transform=ax.transAxes,va='top')
    ax.set_xlabel(r'Opportunity gap $\times 10^5$');ax.set_ylabel(r'OGL $-$ AUD IoU');ax.legend(frameon=False,markerscale=2);ax.set_title('Opportunity does not explain OGL residual gain')
    save(fig,'pool_violation_vs_ogl_gain')
    metrics=['opportunity_gap_mean','violation_fraction_pool','missed_confusing_negative','top4_vs_random4_ratio']
    labels=[r'Opportunity gap ($\times10^5$)','Pool violation','Missed rate','Top4 / random4']
    fig,axes=plt.subplots(1,4,figsize=(10,2.6))
    for ax,m,lab in zip(axes,metrics,labels):
        vals=[x.loc[~a,m].mean(),x.loc[a,m].mean()]
        if m=='opportunity_gap_mean': vals=[v*1e5 for v in vals]
        ax.bar([0,1],vals,color=[COL['non'],COL['groupA']]);ax.set_xticks([0,1],['non-A','A']);ax.set_title(lab,fontsize=8)
    fig.suptitle('VGG-SS 144k: fixed residual Group-A versus non-Group-A',y=1.04)
    save(fig,'groupA_negative_quality')
if __name__=='__main__':main()
