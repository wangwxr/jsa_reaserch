from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
v=pd.read_csv(R/'per_sample/vggss_144k_positive_excess.csv'); f=pd.read_csv(R/'per_sample/flickr_144k_positive_excess.csv')
plt.rcParams.update({'font.size':10,'figure.dpi':160})
def save(fig,name):
 fig.tight_layout();fig.savefig(R/'figures'/f'{name}.png',bbox_inches='tight');fig.savefig(R/'figures'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
pal=['#0072B2','#D55E00','#009E73','#CC79A7']
fig,ax=plt.subplots(figsize=(5,3));ax.boxplot([v[v.groupA].positive_excess_total_mass,v[~v.groupA].positive_excess_total_mass],showfliers=False);ax.set_xticklabels(['VGG Group-A','VGG non-A']);ax.set_ylabel('PositiveExcess total mass');save(fig,'excess_mass_groupA')
fig,ax=plt.subplots(figsize=(5,3));x=v[v.groupA];ax.bar(['RemoveFP','RemoveTP'],[x.excess_precision_removefp.mean(),x.excess_precision_removetp.mean()],color=pal[:2]);ax.set_ylabel('Fraction of excess pixels');ax.set_title('VGG Group-A');save(fig,'excess_removefp_overlap')
fig,ax=plt.subplots(figsize=(5,3));z=pd.DataFrame({'VGG Group-A':[v[v.groupA].excess_gt_ratio.mean(),v[v.groupA].excess_fp_ratio.mean()],'VGG AUD-better':[v[v.aud_better].excess_gt_ratio.mean(),v[v.aud_better].excess_fp_ratio.mean()],'Flickr':[f.excess_gt_ratio.mean(),f.excess_fp_ratio.mean()]},index=['GT','FP']).T;z.plot.bar(stacked=True,color=pal[:2],ax=ax);ax.set_ylim(0,1);ax.set_ylabel('PositiveExcess mass fraction');save(fig,'excess_tp_fp_tradeoff')
fig,ax=plt.subplots(figsize=(5,3));z=pd.DataFrame({'VGG all':v.excess_fp_ratio,'VGG Group-A':v[v.groupA].excess_fp_ratio,'Flickr':f.excess_fp_ratio});ax.boxplot([z[c].dropna() for c in z],showfliers=False);ax.set_xticklabels(z.columns,rotation=12);ax.set_ylabel('Excess FP ratio');save(fig,'vgg_vs_flickr_excess')
