#!/usr/bin/env python3
"""Paired, group, and internal-hardness analysis for the frozen map audit."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE=Path(__file__).resolve().parent; EXP=HERE.parent
BOOT=2000

def stats(x, seed=12345):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if len(x)==0:return {'mean':np.nan,'median':np.nan,'ci_low':np.nan,'ci_high':np.nan,'n':0}
    rng=np.random.default_rng(seed); means=[]
    for _ in range(0,BOOT,100):
        z=rng.integers(len(x),size=(min(100,BOOT-len(means)),len(x)));means.extend(x[z].mean(1))
    return {'mean':float(x.mean()),'median':float(np.median(x)),'ci_low':float(np.quantile(means,.025)),'ci_high':float(np.quantile(means,.975)),'n':len(x)}

def diff_stats(x,y,seed=12345):
    x=np.asarray(x,float);y=np.asarray(y,float);x=x[np.isfinite(x)];y=y[np.isfinite(y)]
    rng=np.random.default_rng(seed);boot=[]
    for _ in range(0,BOOT,100):
        a=x[rng.integers(len(x),size=(min(100,BOOT-len(boot)),len(x)))].mean(1);b=y[rng.integers(len(y),size=(len(a),len(y)))].mean(1);boot.extend(a-b)
    u,p=mannwhitneyu(x,y,alternative='two-sided');rbc=1-2*u/(len(x)*len(y))
    return {'mean_difference':float(x.mean()-y.mean()),'ci_low':float(np.quantile(boot,.025)),'ci_high':float(np.quantile(boot,.975)),'mannwhitney_p':float(p),'rank_biserial':float(rbc),'n_left':len(x),'n_right':len(y)}

def main():
    perf=pd.read_csv(HERE/'performance/all_readouts.csv');six=pd.read_csv(HERE/'performance/six_readout_per_sample.csv');obj=pd.read_csv(HERE/'mechanism/object_per_sample_all.csv');trans=pd.read_csv(HERE/'mechanism/hard_minus_mean_transition_per_sample.csv')
    # Scale-effect table: exact Hard-minus-Mean values at the two data scales.
    rows=[]
    for ds in ('vggss','flickr'):
      for scale in ('10k','144k'):
        p=perf[(perf.scale==scale)&(perf.dataset==ds)&(perf.readout=='AUD')].set_index('model')
        m=obj[(obj.scale==scale)&(obj.dataset==ds)].groupby('model')[['gt_nearfp_gap','gt_vs_nearfp_auroc','coverage','precision','pred_area_ratio']].mean()
        row={'dataset':ds,'scale':scale,'hard_effect_cIoU':p.loc['hard_default','cIoU']-p.loc['mean','cIoU'],'hard_effect_AUC':p.loc['hard_default','AUC']-p.loc['mean','AUC']}
        for col in ['gt_nearfp_gap','gt_vs_nearfp_auroc','coverage','precision','pred_area_ratio']:
          row['hard_effect_'+col]=m.loc['hard_default',col]-m.loc['mean',col]
        rows.append(row)
    pd.DataFrame(rows).to_csv(HERE/'scale_comparison/10k_vs_144k.csv',index=False)
    # Frozen residual groups: do not redefine labels or thresholds.
    residual=pd.read_csv(EXP/'residual_ogl_gap_audit/per_sample/vggss_aud_vs_ogl_per_sample.csv',dtype={'sample_id':str})
    natural=obj[(obj.scale=='144k')&(obj.dataset=='vggss')].merge(residual[['sample_id','group','gt_area_ratio','size_bin','auto_taxonomy','over_diagnosis','delta_iou']],on='sample_id',how='left')
    aud=six[(six.scale=='144k')&(six.dataset=='vggss')&(six.readout=='AUD')][['model','sample_id','iou']].rename(columns={'iou':'aud_iou'})
    ga=[]
    metrics=['iou','gt_response','nearfp_response','gt_nearfp_gap','gt_vs_nearfp_auroc','precision','coverage','pred_area_ratio','RemoveFP','RemoveTP','AddTP','AddFP']
    for model,frame in natural.merge(aud,on=['model','sample_id']).groupby('model'):
      group=frame[frame.group.eq('A_ogl_help')]
      row={'dataset':'vggss','scale':'144k','model':model,'group':'A_ogl_help','n':len(group),'cIoU_success_count':int((group.aud_iou>=.5).sum())}
      for metric in metrics: row[metric]=float(group[metric].mean())
      if model!='mean':
        tt=trans[(trans.scale=='144k')&(trans.dataset=='vggss')&(trans.model==model)][['sample_id','RemoveFP','RemoveTP','AddTP','AddFP']].merge(group[['sample_id']],on='sample_id')
        for metric in ['RemoveFP','RemoveTP','AddTP','AddFP']:row['vs_mean_'+metric]=float(tt[metric].mean())
      ga.append(row)
    pd.DataFrame(ga).to_csv(HERE/'residual_group/groupA_comparison.csv',index=False)
    # Help/hurt: Hard-minus-Mean with predefined numeric thresholds, not re-picked by taxonomy.
    hh=[]
    for ds in ('vggss','flickr'):
      base=six[(six.scale=='144k')&(six.dataset==ds)&(six.readout=='AUD')&(six.model=='mean')][['sample_id','iou']].rename(columns={'iou':'mean_iou'})
      hard=six[(six.scale=='144k')&(six.dataset==ds)&(six.readout=='AUD')&(six.model=='hard_default')][['sample_id','iou']].rename(columns={'iou':'hard_iou'})
      info=obj[(obj.scale=='144k')&(obj.dataset==ds)&(obj.model=='mean')][['sample_id','gt_response','nearfp_response','gt_nearfp_gap','gt_vs_nearfp_auroc','precision','coverage','pred_area_ratio']]
      d=base.merge(hard,on='sample_id').merge(info,on='sample_id');d['hard_minus_mean_iou']=d.hard_iou-d.mean_iou;d['subset']=np.select([d.hard_minus_mean_iou<-.1,d.hard_minus_mean_iou>.1],['hard_hurt','hard_help'],default='middle')
      if ds=='vggss':d=d.merge(residual[['sample_id','gt_area_ratio','size_bin','auto_taxonomy','over_diagnosis','group']],on='sample_id',how='left')
      for subset,frame in d[d.subset!='middle'].groupby('subset'):
        row={'dataset':ds,'subset':subset,'n':len(frame)}
        for col in ['hard_minus_mean_iou','mean_iou','precision','coverage','pred_area_ratio','gt_nearfp_gap','nearfp_response','gt_vs_nearfp_auroc']:
          row[col]=float(frame[col].mean())
        if ds=='vggss': row['gt_area_ratio']=float(frame.gt_area_ratio.mean())
        hh.append(row)
      d.to_csv(HERE/f'residual_group/hard_help_hurt_per_sample_{ds}.csv',index=False)
    pd.DataFrame(hh).to_csv(HERE/'residual_group/hard_help_hurt.csv',index=False)
    # Internal test hardness, joined to residual labels only after computation.
    hrows=[]; allh=[]
    signals=['D_min','D_mean','D_max','D_std','mean_minus_min','pos_minus_min','pos_minus_mean','negative_cv','weight_max','N_eff','weight_entropy']
    for scale in ('10k','144k'):
      for ds in ('vggss','flickr'):
        d=pd.read_csv(HERE/f'hardness_signal/test_distances_{scale}_{ds}.csv',dtype={'sample_id':str})
        d['scale']=scale;d['dataset']=ds
        if scale=='144k' and ds=='vggss':
          d=d.merge(residual[['sample_id','group','delta_iou']],on='sample_id',how='left')
          d['group_a']=d.group.eq('A_ogl_help')
          for label,left,right in [('GroupA_vs_nonA',d[d.group_a],d[~d.group_a]),('OGLhelp_vs_AUDbetter',d[d.group.eq('A_ogl_help')],d[d.group.eq('C_aud_better')])]:
            for signal in signals:hrows.append({'scale':scale,'dataset':ds,'comparison':label,'signal':signal,**diff_stats(left[signal],right[signal],hash((label,signal))%100000)})
        # Constraint diagnostics use D_pos < D_mean; no labels are involved.
        neg=d[[f'd_neg_{i}' for i in range(1,5)]].to_numpy(); corr=np.corrcoef(neg,rowvar=False); off=corr[np.triu_indices(4,1)]
        hrows.append({'scale':scale,'dataset':ds,'comparison':'margin_diagnostic','signal':'violation_pos_ge_mean_fraction',**stats((d.pos_minus_mean>=0).astype(float))})
        hrows.append({'scale':scale,'dataset':ds,'comparison':'margin_diagnostic','signal':'hard_but_mean_easy_fraction',**stats(((d.pos_minus_min>=0)&(d.pos_minus_mean<0)).astype(float))})
        hrows.append({'scale':scale,'dataset':ds,'comparison':'margin_diagnostic','signal':'mean_pairwise_negative_correlation',**stats(off)})
        allh.append(d)
    pd.concat(allh,ignore_index=True).to_csv(HERE/'hardness_signal/per_sample_signals.csv',index=False)
    pd.DataFrame(hrows).to_csv(HERE/'hardness_signal/signal_summary.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(ga).to_string(index=False));print(pd.DataFrame(hrows).to_string(index=False))
if __name__=='__main__':main()
