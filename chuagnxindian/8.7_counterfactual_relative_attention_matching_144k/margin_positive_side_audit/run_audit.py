#!/usr/bin/env python3
"""Read-only Phase A/B audit for frozen Mean-CRAM 8.7.

Phase A reuses exact fixed-set1 K=4 distances previously measured on the
frozen Mean Stage-1 teacher.  Phase B is only executed after the predeclared
Phase-A-negative decision and replays cached maps/features; no model forward,
training, candidate selection, or parameter update occurs.
"""
from __future__ import annotations
import json, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu, spearmanr, rankdata

HERE=Path(__file__).resolve().parent; EXP=HERE.parent; ROOT=HERE.parents[2]
XSA=EXP/'cross_scale_cram_audit'; RES=EXP/'residual_ogl_gap_audit'; NQA=EXP/'negative_quality_audit'
E80=ROOT/'chuagnxindian/8.0_audio_only_residual_audit'
# Fixed 300-resample bootstrap.  Point estimates, nonparametric tests, and all
# sample-level data are exact; this only controls Monte-Carlo precision of CIs.
NBOOT=300; EPS=1e-12

def rng(key): return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little'))
def arr(x): return pd.to_numeric(pd.Series(x),errors='coerce').dropna().to_numpy(float)
def desc(x,key):
    x=arr(x); r=rng('d:'+key)
    if len(x)==0:return {'n':0}
    means=np.array([x[r.integers(0,len(x),len(x))].mean() for _ in range(NBOOT)])
    return {'n':int(len(x)),'mean':float(x.mean()),'median':float(np.median(x)),'p10':float(np.quantile(x,.1)),'p25':float(np.quantile(x,.25)),'p75':float(np.quantile(x,.75)),'p90':float(np.quantile(x,.9)),'bootstrap_mean_ci_low':float(np.quantile(means,.025)),'bootstrap_mean_ci_high':float(np.quantile(means,.975))}
def comp(a,b,key):
    a,b=arr(a),arr(b); out={'left':desc(a,key+'l'),'right':desc(b,key+'r')}
    r=rng('c:'+key); ds=np.array([a[r.integers(0,len(a),len(a))].mean()-b[r.integers(0,len(b),len(b))].mean() for _ in range(NBOOT)])
    u,p=mannwhitneyu(a,b,alternative='two-sided');out.update({'mean_diff_left_minus_right':float(a.mean()-b.mean()),'bootstrap_ci_low':float(np.quantile(ds,.025)),'bootstrap_ci_high':float(np.quantile(ds,.975)),'mannwhitney_p':float(p),'rank_biserial':float(2*u/(len(a)*len(b))-1)})
    return out
def flatten(title, dd):
    rows=[]
    for metric,v in dd.items():
        row={'comparison':title,'metric':metric}
        for side in ('left','right'):
            for k,z in v[side].items():row[f'{side}_{k}']=z
        for k in ('mean_diff_left_minus_right','bootstrap_ci_low','bootstrap_ci_high','mannwhitney_p','rank_biserial'):row[k]=v.get(k,np.nan)
        rows.append(row)
    return rows
def bootstrap_rho(x,y,key):
    x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok];r=rng('rho:'+key)
    # Bootstrap the correlation of the fixed full-sample ranks. The point
    # estimate remains scipy's exact Spearman; with continuous distances this
    # is the standard efficient percentile-bootstrap approximation.
    rx=rankdata(x);ry=rankdata(y);vals=[]
    for _ in range(NBOOT):
        i=r.integers(0,len(x),len(x));vals.append(float(np.corrcoef(rx[i],ry[i])[0,1]))
    rho,p=spearmanr(x,y)
    return {'n':int(len(x)),'spearman_rho':float(rho),'p_value':float(p),'bootstrap_ci_low':float(np.quantile(vals,.025)),'bootstrap_ci_high':float(np.quantile(vals,.975))}
def normalize(v):
    lo=v.amin(dim=(-2,-1),keepdim=True);hi=v.amax(dim=(-2,-1),keepdim=True);return (v-lo)/(hi-lo).clamp_min(EPS)
def resize_norm(values,device):
    size=int(round(values.shape[-1]**.5));t=torch.from_numpy(values).to(device=device,dtype=torch.float32).reshape(-1,1,size,size);return normalize(F.interpolate(t,(224,224),mode='bicubic',align_corners=False)[:,0]).cpu().numpy()
def region_metrics(score,gt,fixed,index,compute_entropy=False):
    gtsoft=gt[index]; gtb=gtsoft>=.5; flat=score.ravel();gi=fixed['indices'][index,0,:int(fixed['counts'][index,0])];ni=fixed['indices'][index,2,:int(fixed['counts'][index,2])]
    pred=score>=.6;inter=float((pred*gtsoft).sum());total=float(score.sum())
    bg=score[~gtb]
    entropy=float(-(np.clip(score,EPS,1)*np.log(np.clip(score,EPS,1))+(1-np.clip(score,EPS,1))*np.log(np.clip(1-score,EPS,1))).mean()) if compute_entropy else np.nan
    return {'gt_response':float(flat[gi].mean()) if len(gi) else np.nan,'nearfp_response':float(flat[ni].mean()) if len(ni) else np.nan,'background_response':float(bg.mean()),'gt_nearfp_gap':float(flat[gi].mean()-flat[ni].mean()) if len(gi) and len(ni) else np.nan,'gt_context_ratio':float(flat[gi].mean()/max(bg.mean(),EPS)) if len(gi) else np.nan,'precision':inter/max(float(pred.sum()),EPS),'coverage':inter/max(float(gtsoft.sum()),EPS),'activated_area':float(pred.mean()),'entropy':entropy,'top_response_area_ratio':float((score>=.8).mean()),'response_mass_inside_gt':float((score*gtsoft).sum()/max(total,EPS)),'response_mass_outside_gt':float((score*(1-gtsoft)).sum()/max(total,EPS)),'outside_inside_response_ratio':float((score*(1-gtsoft)).sum()/max((score*gtsoft).sum(),EPS))}

def phase_a():
    signals=pd.read_csv(XSA/'hardness_signal/per_sample_signals.csv')
    residual=pd.read_csv(RES/'per_sample/vggss_aud_vs_ogl_per_sample.csv')
    all_frames=[]
    # Reconstruct natural Mean-CRAM margin from exact fixed 8.1 set1 K=4 distances.
    for scale in ['10k','144k']:
        for dataset in ['vggss','flickr']:
            x=signals[(signals.scale.eq(scale))&(signals.dataset.eq(dataset))].copy()
            x['d_neg_mean']=x.D_mean;x['d_neg_min']=x.D_min;x['d_neg_max']=x.D_max;x['margin']=x.D_mean-x.d_pos;x['violation']=x.margin.le(0);x['softplus_loss']=np.logaddexp(0,-x.margin);x['normalized_margin']=x.margin/(x.D_mean.abs()+x.d_pos.abs()+EPS)
            all_frames.append(x)
    allx=pd.concat(all_frames,ignore_index=True)
    v=allx[(allx.scale.eq('144k'))&(allx.dataset.eq('vggss'))].merge(residual,on='sample_id',how='left',suffixes=('','_res'))
    assert len(v)==5158 and v.group.eq('A_ogl_help').sum()==428
    # Store requested evaluation-sample quantities plus frozen diagnosis fields.
    cols=['sample_id','sample_index','d_pos','d_neg_mean','d_neg_min','d_neg_max','margin','violation','softplus_loss','normalized_margin','group','delta_iou','aud_iou','ogl_iou','aud_precision','aud_coverage','aud_area','aud_gap','aud_auroc','aud_nearfp_response','auto_taxonomy']
    v[cols].to_csv(HERE/'margin/per_sample_margin.csv',index=False)
    a=v.group.eq('A_ogl_help'); ms=['margin','violation','softplus_loss','normalized_margin']
    gc={m:comp(v.loc[a,m],v.loc[~a,m],f'groupA:{m}') for m in ms}
    pd.DataFrame(flatten('fixed Group-A vs non-Group-A',gc)).to_csv(HERE/'margin/groupA_margin_summary.csv',index=False)
    # Residual association including taxonomy as an explicit OVER/non-OVER comparison.
    cr=[]
    for m in ['margin','violation','softplus_loss','normalized_margin']:
        for y in ['delta_iou','aud_precision','aud_area','aud_gap','aud_nearfp_response']:
            cr.append({'margin_metric':m,'outcome':y,**bootstrap_rho(v[m],v[y],m+y)})
    pd.DataFrame(cr).to_csv(HERE/'margin/margin_correlations.csv',index=False)
    over={m:comp(v.loc[v.auto_taxonomy.eq('OVER'),m],v.loc[~v.auto_taxonomy.eq('OVER'),m],f'over:{m}') for m in ms}
    pd.DataFrame(flatten('frozen OVER taxonomy vs non-OVER',over)).to_csv(HERE/'margin/margin_over_taxonomy.csv',index=False)
    v['margin_bucket']=pd.qcut(v.margin,5,labels=['Q1_lowest','Q2','Q3','Q4','Q5_highest'],duplicates='drop')
    bucket=v.groupby('margin_bucket',observed=False).agg(n=('sample_id','size'),margin_mean=('margin','mean'),groupA_fraction=('group',lambda z:float((z=='A_ogl_help').mean())),ogl_minus_aud_iou=('delta_iou','mean'),precision=('aud_precision','mean'),coverage=('aud_coverage','mean'),activated_area=('aud_area','mean'),gt_nearfp_gap=('aud_gap','mean'),aud_iou=('aud_iou','mean'),violation_rate=('violation','mean')).reset_index()
    bucket.to_csv(HERE/'margin/margin_bucket_summary.csv',index=False)
    control=allx.groupby(['scale','dataset']).agg(n=('sample_id','size'),margin_mean=('margin','mean'),margin_median=('margin','median'),violation_rate=('violation','mean'),softplus_loss_mean=('softplus_loss','mean'),normalized_margin_mean=('normalized_margin','mean')).reset_index()
    control.to_csv(HERE/'margin/margin_control_summary.csv',index=False)
    # Predeclared A-positive requires all three directional evidence components.
    g=gc['margin']; vv=gc['violation']; rho=[z for z in cr if z['margin_metric']=='margin' and z['outcome']=='delta_iou'][0]
    positive=(g['bootstrap_ci_high']<0 and vv['bootstrap_ci_low']>0 and rho['spearman_rho']<-.1 and rho['bootstrap_ci_high']<0)
    decision={'phase_A':'A-POSITIVE' if positive else 'A-NEGATIVE','rule':'Group-A margin CI wholly below 0 AND Group-A violation CI wholly above 0 AND rho(margin,OGL-AUD)<-0.1 with CI wholly below 0','observed_margin_diff':g['mean_diff_left_minus_right'],'observed_violation_diff':vv['mean_diff_left_minus_right'],'observed_rho_margin_ogl_gain':rho}
    (HERE/'margin/phase_a_decision.json').write_text(json.dumps(decision,indent=2))
    return v,decision

def phase_b(v):
    # This function is invoked only after A-NEGATIVE. It uses maps already
    # cached by mechanism/residual audits, not model outputs recomputed here.
    maps=np.load(EXP/'mechanism_audit/results/vggss/full/cram_c3_stage2_maps.npz',allow_pickle=False); nat=np.load(EXP/'natural_localization/vggss/C3/seed12345/natural_maps.npz',allow_pickle=False);prior=np.load(EXP/'stage_specific_audit/degradation_path/vggss/object_prior.npz',allow_pickle=False);fixed=np.load(E80/'results/vggss_144k_sample_indices.npz',allow_pickle=False)
    ids=maps['ids'].astype(str); assert np.array_equal(ids,nat['ids'].astype(str)) and np.array_equal(ids,fixed['ids'].astype(str)) and np.array_equal(ids,prior['ids'].astype(str))
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');gt=nat['gt_masks'].astype(np.float32);rows=[];trans=[];read=[]
    # Process maps in compact chunks to keep this audit well below GPU memory.
    for st in range(0,len(ids),128):
        en=min(st+128,len(ids)); aud=resize_norm(maps['H_matched'][st:en],device);img=resize_norm(nat['image_native'][st:en].reshape(en-st,49),device);pr=resize_norm(prior['maps'][st:en],device)
        iqr=normalize(torch.from_numpy(.6*aud+.4*img)).numpy();ogl=normalize(torch.from_numpy(.6*aud+.4*pr)).numpy()
        for local,i in enumerate(range(st,en)):
            metric_by={}
            for name,score in [('AUD',aud[local]),('IMG_QUERY',img[local]),('IQR',iqr[local]),('OGL',ogl[local])]:
                z=region_metrics(score,gt,fixed,i,compute_entropy=name=='AUD');metric_by[name]=z;read.append({'sample_id':ids[i],'sample_index':i,'readout':name,**z})
            # OGL correction is described against the matched AUD map by global
            # and NearFP-region transitions, not by counterfactual selection.
            ap=aud[local]>=.6;op=ogl[local]>=.6;g=gt[i]>=.5;rem=ap&~op;add=op&~ap;near=np.zeros((224*224,),bool);ni=fixed['indices'][i,2,:int(fixed['counts'][i,2])];near[ni]=True;near=near.reshape(224,224)
            trans.append({'sample_id':ids[i],'sample_index':i,'ogl_minus_aud_gt_response':metric_by['OGL']['gt_response']-metric_by['AUD']['gt_response'],'ogl_minus_aud_nearfp_response':metric_by['OGL']['nearfp_response']-metric_by['AUD']['nearfp_response'],'ogl_minus_aud_background_response':metric_by['OGL']['background_response']-metric_by['AUD']['background_response'],'RemoveFP':int((rem&~g).sum()),'RemoveTP':int((rem&g).sum()),'AddTP':int((add&g).sum()),'AddFP':int((add&~g).sum()),'RemoveNearFP':int((rem&near).sum()),'AddNearFP':int((add&near).sum())})
    readdf=pd.DataFrame(read);audit=v.merge(readdf[readdf.readout.eq('AUD')].drop(columns='readout'),on=['sample_id','sample_index'],suffixes=('','_map'),validate='one_to_one')
    # Cached residual values validate primary AUD metrics; map-replay metrics
    # add mass/ratio statistics that were not previously materialized.
    audit.to_csv(HERE/'positive_map/per_sample_positive_metrics.csv',index=False)
    a=audit.group.eq('A_ogl_help');pm=['gt_response','nearfp_response','background_response','gt_nearfp_gap','gt_context_ratio','activated_area','precision','coverage','entropy','top_response_area_ratio','response_mass_inside_gt','response_mass_outside_gt','outside_inside_response_ratio']
    gp={m:comp(audit.loc[a,m],audit.loc[~a,m],f'positive:{m}') for m in pm}
    pd.DataFrame(flatten('fixed Group-A vs non-Group-A positive AUD map',gp)).to_csv(HERE/'positive_map/groupA_positive_summary.csv',index=False)
    td=pd.DataFrame(trans);td.to_csv(HERE/'positive_map/ogl_correction_per_sample.csv',index=False)
    tc={m:desc(td.loc[a,m],f'oglA:{m}') for m in ['ogl_minus_aud_gt_response','ogl_minus_aud_nearfp_response','ogl_minus_aud_background_response','RemoveFP','RemoveTP','AddTP','AddFP','RemoveNearFP','AddNearFP']}
    pd.DataFrame([{'metric':m,**z} for m,z in tc.items()]).to_csv(HERE/'positive_map/groupA_ogl_correction_summary.csv',index=False)
    # Four-way partition: spatial OVER criterion is the pre-existing frozen taxonomy.
    audit['constraint_state']=np.where(audit.margin>0,'satisfied','violated');audit['spatial_state']=np.where(audit.auto_taxonomy.eq('OVER'),'over_activated','not_over');four=audit.groupby(['constraint_state','spatial_state']).agg(n=('sample_id','size'),groupA_n=('group',lambda z:int((z=='A_ogl_help').sum())),groupA_fraction=('group',lambda z:float((z=='A_ogl_help').mean())),mean_ogl_minus_aud=('delta_iou','mean'),precision=('precision','mean'),coverage=('coverage','mean'),activated_area=('activated_area','mean')).reset_index();four.to_csv(HERE/'satisfied_but_over/four_way_partition.csv',index=False)
    # For requested Group-A focus, compare AUD / IMG_QUERY / IQR without OGL
    # selection; these are all deterministic cached readouts.
    rr=readdf.merge(audit[['sample_id','group']],on='sample_id',validate='many_to_one');rra=rr[rr.group.eq('A_ogl_help') & rr.readout.isin(['AUD','IMG_QUERY','IQR'])];summary=rra.groupby('readout').agg(n=('sample_id','size'),gt_response=('gt_response','mean'),nearfp_response=('nearfp_response','mean'),gt_nearfp_gap=('gt_nearfp_gap','mean'),precision=('precision','mean'),coverage=('coverage','mean'),activated_area=('activated_area','mean'),response_mass_inside_gt=('response_mass_inside_gt','mean'),response_mass_outside_gt=('response_mass_outside_gt','mean')).reset_index();summary.to_csv(HERE/'readout_comparison/aud_img_iqr_groupA.csv',index=False)
    # C3 visual-key similarity: pool the existing 7x7 feature tokens over GT
    # and frozen NearFP pixel regions. This is descriptive, never used for a loss.
    keys=np.load(EXP/'natural_localization/vggss/C3/seed12345/visual_keys.npy',mmap_mode='r');sim=[]
    for i in range(len(ids)):
        g7=(gt[i].reshape(7,32,7,32).mean((1,3))).reshape(49);near=np.zeros(224*224,float);ni=fixed['indices'][i,2,:int(fixed['counts'][i,2])];near[ni]=1.;n7=near.reshape(7,32,7,32).mean((1,3)).reshape(49)
        gv=(keys[i]*g7[:,None]).sum(0)/max(g7.sum(),EPS);nv=(keys[i]*n7[:,None]).sum(0)/max(n7.sum(),EPS);cos=float(np.dot(gv,nv)/max(np.linalg.norm(gv)*np.linalg.norm(nv),EPS));sim.append({'sample_id':ids[i],'sample_index':i,'gt_nearfp_visual_cosine':cos,'gt_token_mass':float(g7.sum()),'nearfp_token_mass':float(n7.sum())})
    sim=pd.DataFrame(sim).merge(audit[['sample_id','group']],on='sample_id',validate='one_to_one');sim.to_csv(HERE/'visual_similarity/gt_nearfp_feature_similarity.csv',index=False);sc=comp(sim.loc[sim.group.eq('A_ogl_help'),'gt_nearfp_visual_cosine'],sim.loc[~sim.group.eq('A_ogl_help'),'gt_nearfp_visual_cosine'],'visual_similarity');pd.DataFrame(flatten('fixed Group-A vs non-Group-A C3 visual cosine',{'gt_nearfp_visual_cosine':sc})).to_csv(HERE/'visual_similarity/groupA_visual_similarity_summary.csv',index=False)
    return {'positive_group_comparison':gp,'four_way':four.to_dict('records'),'readout_groupA':summary.to_dict('records'),'visual_similarity':sc,'ogl_groupA':tc}

def main():
    for d in ['margin','positive_map','satisfied_but_over','readout_comparison','visual_similarity','figures','summary']: (HERE/d).mkdir(parents=True,exist_ok=True)
    v,decision=phase_a();out={'phase_a':decision}
    if decision['phase_A']=='A-NEGATIVE':out['phase_b']=phase_b(v)
    else:out['phase_b']='not_run: Phase A positive'
    (HERE/'summary/analysis_bundle.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
