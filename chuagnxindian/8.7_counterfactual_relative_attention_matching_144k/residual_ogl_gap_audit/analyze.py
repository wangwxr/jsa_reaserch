#!/usr/bin/env python3
"""Read-only VGG residual OGL-gap audit for the frozen 8.7 vanilla Stage-2.

Replays the established AUD/OGL evaluation exactly from frozen saved maps.  It
does not load a trainable model and does not alter any prior experiment output.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import label
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
OLD = EXP / "stage_specific_audit/degradation_path/vggss"
MECH = EXP / "mechanism_audit"
SAMPLE = ROOT / "chuagnxindian/8.0_audio_only_residual_audit/results/vggss_144k_sample_indices.npz"
D2 = ROOT / "chuagnxindian/1.3G诊断/d2OGL Mechanism Decomposition/results/per_sample_index.csv"
MANUAL = ROOT / "chuagnxindian/1.3G诊断/d1.6Failure Factor Probe/input/manual_review_merged.csv"
SEED = 870715
NBOOT = 2000
THRESHOLD = 0.6
GROUPS = {"A_ogl_help": (0.10, math.inf), "B_close": (-0.05, 0.05), "C_aud_better": (-math.inf, -0.10)}

def seed(*parts: str) -> int:
    raw = "::".join(map(str, (SEED, *parts))).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % 2**32

def normalize(x: np.ndarray) -> np.ndarray:
    low=x.min(axis=(-2,-1),keepdims=True); high=x.max(axis=(-2,-1),keepdims=True)
    return (x-low)/np.maximum(high-low,1e-12)

def upsample(x: np.ndarray, device: torch.device) -> np.ndarray:
    n=x.shape[0]; s=int(round(math.sqrt(x.shape[-1])))
    chunks=[]
    for start in range(0,n,128):
        t=torch.from_numpy(x[start:start+128]).to(device).reshape(-1,1,s,s).float()
        chunks.append(F.interpolate(t,(224,224),mode="bicubic",align_corners=False)[:,0].cpu().numpy())
    return np.concatenate(chunks)

def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    return float((pred*gt).sum())/max(float(gt.sum()+np.logical_and(pred,gt==0).sum()),1e-12)

def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if not len(pos) or not len(neg): return math.nan
    r=rankdata(np.r_[pos,neg],method="average"); return float((r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))

def summarise(x: np.ndarray, *parts: str) -> dict:
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if not len(x): return dict(mean=math.nan,median=math.nan,ci_low=math.nan,ci_high=math.nan,valid=0)
    rng=np.random.default_rng(seed(*parts)); boot=[]
    for _ in range(0,NBOOT,100):
        ind=rng.integers(0,len(x),(min(100,NBOOT-len(boot)),len(x))); boot.extend(x[ind].mean(1))
    return dict(mean=float(x.mean()),median=float(np.median(x)),q1=float(np.quantile(x,.25)),q3=float(np.quantile(x,.75)),
                ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),valid=int(len(x)))

def corr(x: np.ndarray,y: np.ndarray,method: str) -> float:
    keep=np.isfinite(x)&np.isfinite(y)
    if keep.sum()<3 or np.std(x[keep])==0 or np.std(y[keep])==0:return math.nan
    return float(spearmanr(x[keep],y[keep]).statistic if method=="spearman" else np.corrcoef(x[keep],y[keep])[0,1])

def json_safe(value):
    if isinstance(value, dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value, list): return [json_safe(v) for v in value]
    if isinstance(value, (float,np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (int,np.integer)): return int(value)
    return value

def main() -> None:
    for output_dir in (HERE/'per_sample',HERE/'group_analysis',HERE/'ogl_gain_decomposition',HERE/'figures',HERE/'summary'):
        output_dir.mkdir(parents=True,exist_ok=True)
    dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with np.load(EXP/'natural_localization/vggss/C3/seed12345/natural_maps.npz') as z:
        ids=z['ids'].astype(str); gt=z['gt_masks'].astype(np.float32)
    with np.load(MECH/'results/vggss/full/cram_c3_stage2_maps.npz') as z:
        assert np.array_equal(ids,z['ids'].astype(str)); aud_native=z['H_matched'].astype(np.float32)
    with np.load(OLD/'object_prior.npz') as z:
        assert np.array_equal(ids,z['ids'].astype(str)); prior_native=z['maps'].astype(np.float32)
    with np.load(SAMPLE) as z: fixed={k:z[k].copy() for k in z.files}
    assert np.array_equal(ids,fixed['ids'].astype(str))
    aud=normalize(upsample(aud_native,dev)); prior=normalize(upsample(prior_native,dev)); ogl=normalize(.6*aud+.4*prior)
    cached=pd.read_csv(OLD/'six_readout_per_sample.csv',dtype={'sample_id':str})
    cached=cached[cached.model.eq('cram_c3_stage2')].set_index('sample_id').loc[ids]
    # Vectorize full-image quantities; only the variable-size sampled GT/NearFP
    # regions below require an image loop.
    ap=aud>=THRESHOLD; op=ogl>=THRESHOLD; gb=gt>=.5
    def full_metrics(score,pred):
        inter=(pred*gt).sum((1,2)); denom=gt.sum((1,2))+np.logical_and(pred,gt==0).sum((1,2))
        mass=score/np.maximum(score.sum((1,2),keepdims=True),1e-30)
        return dict(iou=inter/np.maximum(denom,1e-12),coverage=inter/np.maximum(gt.sum((1,2)),1e-12),
                    precision=inter/np.maximum(pred.sum((1,2)),1e-12),area=pred.mean((1,2)),
                    entropy=(-(mass*np.log(np.maximum(mass,1e-30))).sum((1,2))/np.log(score.shape[-1]*score.shape[-2])))
    af=full_metrics(aud,ap);of=full_metrics(ogl,op)
    add=op&~ap; remove=ap&~op
    rows=[]
    for i,sid in enumerate(ids):
        g=gt[i]
        gi=fixed['indices'][i,0,:int(fixed['counts'][i,0])]; ni=fixed['indices'][i,2,:int(fixed['counts'][i,2])]
        av=aud[i].ravel();ov=ogl[i].ravel()
        row=dict(sample_id=sid,sample_index=i,gt_area_ratio=float(gb[i].mean()),gt_components=int(label(gb[i])[1]),
                 AddTP=int((add[i]&gb[i]).sum()),AddFP=int((add[i]&~gb[i]).sum()),RemoveTP=int((remove[i]&gb[i]).sum()),RemoveFP=int((remove[i]&~gb[i]).sum()))
        for prefix,values,whole in [('aud',av,af),('ogl',ov,of)]:
            row.update({f'{prefix}_iou':float(whole['iou'][i]),f'{prefix}_gt_response':float(values[gi].mean()) if len(gi) else math.nan,
                f'{prefix}_nearfp_response':float(values[ni].mean()) if len(ni) else math.nan,
                f'{prefix}_gap':float(values[gi].mean()-values[ni].mean()) if len(gi) and len(ni) else math.nan,
                f'{prefix}_auroc':auc(values[gi],values[ni]),f'{prefix}_coverage':float(whole['coverage'][i]),
                f'{prefix}_precision':float(whole['precision'][i]),f'{prefix}_area':float(whole['area'][i]),f'{prefix}_entropy':float(whole['entropy'][i])})
        row['delta_iou']=row['ogl_iou']-row['aud_iou']
        rows.append(row)
    df=pd.DataFrame(rows)
    aud_cache_error=float(np.max(np.abs(df.aud_iou.to_numpy()-cached.AUD.to_numpy())))
    ogl_cache_error=float(np.max(np.abs(df.ogl_iou.to_numpy()-cached.OGL.to_numpy())))
    assert aud_cache_error<1e-6 and ogl_cache_error<1e-6
    df['group']='transition'
    df.loc[df.delta_iou>.10,'group']='A_ogl_help';df.loc[df.delta_iou.between(-.05,.05,inclusive='both'),'group']='B_close';df.loc[df.delta_iou<-.10,'group']='C_aud_better'
    d2=pd.read_csv(D2,dtype={'sample_id':str});d2=d2[d2.dataset.eq('vggss')]
    df=df.merge(d2[['sample_id','auto_taxonomy','spatial_error','audio_condition','over_diagnosis','size_bin','size_quartile']],on='sample_id',how='left',validate='one_to_one')
    manual=pd.read_csv(MANUAL,dtype={'sample_id':str},keep_default_na=False);manual=manual[manual.dataset.eq('vggss')]
    cols=['sample_id','spatial_error','manual_notes'];manual=manual[[x for x in cols if x in manual]].drop_duplicates('sample_id')
    df=df.merge(manual,on='sample_id',how='left',suffixes=('','_manual'))
    df.to_csv(HERE/'per_sample/vggss_aud_vs_ogl_per_sample.csv',index=False)
    metric_cols=['delta_iou','aud_iou','ogl_iou','aud_gt_response','ogl_gt_response','aud_nearfp_response','ogl_nearfp_response','aud_gap','ogl_gap','aud_auroc','ogl_auroc','aud_coverage','ogl_coverage','aud_precision','ogl_precision','aud_area','ogl_area','aud_entropy','ogl_entropy','AddTP','AddFP','RemoveTP','RemoveFP','gt_area_ratio','gt_components']
    gs=[]
    for group,x in df.groupby('group',sort=False):
        r={'group':group,'n':len(x),'fraction':len(x)/len(df)}
        for col in metric_cols:r.update({f'{col}_{k}':v for k,v in summarise(x[col].to_numpy(),group,col).items()})
        gs.append(r)
    group=pd.DataFrame(gs);group.to_csv(HERE/'group_analysis/group_summary.csv',index=False)
    correlations=[]
    for col in ['RemoveFP','AddTP','RemoveTP','AddFP','aud_coverage','ogl_coverage','aud_precision','ogl_precision','aud_gap','ogl_gap','gt_area_ratio','gt_components']:
        correlations.append(dict(metric=col,pearson=corr(df.delta_iou.to_numpy(),df[col].to_numpy(),'pearson'),spearman=corr(df.delta_iou.to_numpy(),df[col].to_numpy(),'spearman')))
    pd.DataFrame(correlations).to_csv(HERE/'ogl_gain_decomposition/delta_iou_correlations.csv',index=False)
    # Per-group paired AUD->OGL changes; direct response metrics make the source of OGL gain explicit.
    paired=[]
    for group_name,x in df.groupby('group',sort=False):
        for base,new,label_name in [('gt_response','gt_response','GT response'),('nearfp_response','nearfp_response','NearFP response'),('gap','gap','GT-Near gap'),('auroc','auroc','GT/Near AUROC'),('coverage','coverage','Coverage'),('precision','precision','Precision'),('area','area','Activated area'),('entropy','entropy','Normalized entropy')]:
            values=x['ogl_'+new].to_numpy()-x['aud_'+base].to_numpy()
            paired.append(dict(group=group_name,metric=label_name,**summarise(values,group_name,label_name)))
    pd.DataFrame(paired).to_csv(HERE/'ogl_gain_decomposition/paired_aud_to_ogl_changes.csv',index=False)
    top=df.nlargest(30,'delta_iou');bottom=df.nsmallest(30,'delta_iou')
    top.to_csv(HERE/'group_analysis/top_ogl_help_samples.csv',index=False);bottom.to_csv(HERE/'group_analysis/top_aud_better_samples.csv',index=False)
    positive=df.delta_iou.clip(lower=0);negative=(-df.delta_iou.clip(upper=0))
    contrib=dict(n=len(df),mean_delta=float(df.delta_iou.mean()),sum_positive=float(positive.sum()),sum_negative=float(negative.sum()),
                 A_positive_share=float(positive[df.group.eq('A_ogl_help')].sum()/max(positive.sum(),1e-12)),
                 A_net_share=float(df.loc[df.group.eq('A_ogl_help'),'delta_iou'].sum()/df.delta_iou.sum()),
                 positive_sample_fraction=float((df.delta_iou>0).mean()))
    # Diagnostic-only manual labels are sparse; preserve missingness rather than treating it as a class.
    taxonomy=[]
    for group_name,x in df.groupby('group',sort=False):
        for col in ['auto_taxonomy','spatial_error','size_bin','size_quartile']:
            for value,count in x[col].fillna('missing').value_counts().items():taxonomy.append(dict(group=group_name,field=col,value=value,count=int(count),fraction=float(count/len(x))))
    pd.DataFrame(taxonomy).to_csv(HERE/'group_analysis/metadata_composition.csv',index=False)
    # Two compact figures avoid cherry-picked heatmaps.
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid')
    fig,ax=plt.subplots(1,2,figsize=(10,3.7));ax[0].hist(df.delta_iou,bins=70,color='#4276a5');ax[0].axvline(0,color='k');ax[0].axvline(.1,color='#b03a2e',ls='--');ax[0].axvline(-.1,color='#b03a2e',ls='--');ax[0].set(xlabel='OGL IoU − AUD IoU',ylabel='samples')
    for name,x in df.groupby('group',sort=False):ax[1].scatter(x.aud_coverage,x.delta_iou,s=5,alpha=.35,label=name)
    ax[1].axhline(0,color='k');ax[1].set(xlabel='AUD coverage',ylabel='OGL IoU − AUD IoU');ax[1].legend(markerscale=2,fontsize=7);fig.tight_layout();fig.savefig(HERE/'figures/vggss_residual_gap_distribution.png',dpi=180);plt.close(fig)
    summary=dict(protocol=dict(dataset='vggss',model='8.7 C3 Stage-1 + vanilla Stage-2',threshold=THRESHOLD,
                 groups={'A_ogl_help':'delta_iou > 0.10','B_close':'-0.05 <= delta_iou <= 0.05','C_aud_better':'delta_iou < -0.10','transition':'remaining values'},bootstraps=NBOOT,seed=SEED),
                 validation=dict(aud_iou_cache_max_abs_error=aud_cache_error,ogl_iou_cache_max_abs_error=ogl_cache_error,n=len(df)),contribution=contrib,
                 overall={k:summarise(df[k].to_numpy(),'overall',k) for k in metric_cols},groups=group.to_dict('records'),correlations=correlations)
    (HERE/'summary/residual_ogl_gap_summary.json').write_text(json.dumps(json_safe(summary),indent=2,allow_nan=False)+'\n')
    overall=pd.DataFrame([{'group':'ALL','n':len(df),'fraction':1,**{col:summarise(df[col].to_numpy(),'ALL',col)['mean'] for col in metric_cols}}])
    pd.concat([overall,group],ignore_index=True,sort=False).to_csv(HERE/'summary/residual_ogl_gap_summary.csv',index=False)
    print(json.dumps(dict(validation=summary['validation'],contribution=contrib,groups=[{k:r[k] for k in ('group','n','fraction','delta_iou_mean','delta_iou_median','aud_iou_mean','ogl_iou_mean')} for r in summary['groups']]),indent=2))

if __name__=='__main__':main()
