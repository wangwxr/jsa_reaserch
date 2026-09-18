#!/usr/bin/env python3
"""Read-only AUD-versus-IMG_QUERY positive-excess audit for 8.7 Mean-CRAM."""
from __future__ import annotations
import json, hashlib, importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

HERE=Path(__file__).resolve().parent; EXP=HERE.parent; ROOT=HERE.parents[2]
XSA=EXP/'cross_scale_cram_audit'; RES=EXP/'residual_ogl_gap_audit'; E80=ROOT/'chuagnxindian/8.0_audio_only_residual_audit'
EPS=1e-12; NBOOT=300; EXCESS_AREA_THRESHOLD=.10
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def seed(k):return int.from_bytes(hashlib.sha256(k.encode()).digest()[:8],'little')
def norm(x):
 lo=x.amin(dim=(-2,-1),keepdim=True);hi=x.amax(dim=(-2,-1),keepdim=True);return (x-lo)/(hi-lo).clamp_min(EPS)
def resize_norm(x,device):
 z=torch.from_numpy(x).to(device=device,dtype=torch.float32);s=int(round(x.shape[-1]**.5));return norm(F.interpolate(z.reshape(-1,1,s,s),(224,224),mode='bicubic',align_corners=False)[:,0]).cpu().numpy()
def desc(x,k):
 x=pd.to_numeric(pd.Series(x),errors='coerce').dropna().to_numpy(float);r=np.random.default_rng(seed('d'+k));means=[x[r.integers(0,len(x),len(x))].mean() for _ in range(NBOOT)]
 return {'n':len(x),'mean':float(x.mean()),'median':float(np.median(x)),'p25':float(np.quantile(x,.25)),'p75':float(np.quantile(x,.75)),'bootstrap_mean_ci_low':float(np.quantile(means,.025)),'bootstrap_mean_ci_high':float(np.quantile(means,.975))}
def compare(a,b,k):
 a=pd.to_numeric(pd.Series(a),errors='coerce').dropna().to_numpy(float);b=pd.to_numeric(pd.Series(b),errors='coerce').dropna().to_numpy(float);r=np.random.default_rng(seed('c'+k));ds=[a[r.integers(0,len(a),len(a))].mean()-b[r.integers(0,len(b),len(b))].mean() for _ in range(NBOOT)];u,p=mannwhitneyu(a,b,alternative='two-sided')
 return {'left':desc(a,k+'l'),'right':desc(b,k+'r'),'mean_diff_left_minus_right':float(a.mean()-b.mean()),'bootstrap_ci_low':float(np.quantile(ds,.025)),'bootstrap_ci_high':float(np.quantile(ds,.975)),'mannwhitney_p':float(p),'rank_biserial':float(2*u/(len(a)*len(b))-1)}
def flat(title,d):
 rows=[]
 for m,z in d.items():
  row={'comparison':title,'metric':m}
  for side in ['left','right']:
   for k,v in z[side].items():row[f'{side}_{k}']=v
  for k in ['mean_diff_left_minus_right','bootstrap_ci_low','bootstrap_ci_high','mannwhitney_p','rank_biserial']:row[k]=z[k]
  rows.append(row)
 return rows
def aucap(score,pos,neg):
 y=np.r_[np.ones(pos.sum(),int),np.zeros(neg.sum(),int)];s=np.r_[score[pos],score[neg]]
 if not pos.any() or not neg.any():return np.nan,np.nan
 return float(roc_auc_score(y,s)),float(average_precision_score(y,s))
def sources(scale,dataset):
 if scale=='144k':
  m=np.load(EXP/f'mechanism_audit/results/{dataset}/full/cram_c3_stage2_maps.npz',allow_pickle=False);n=np.load(EXP/f'natural_localization/{dataset}/C3/seed12345/natural_maps.npz',allow_pickle=False);image=n['image_native'].reshape(len(m['ids']),49);gt=n['gt_masks'];return m['ids'].astype(str),m['H_matched'],image,gt
 p=XSA/f'raw_maps/{scale}/{dataset}/mean/maps.npz';z=np.load(p,allow_pickle=False);n=np.load(EXP/f'natural_localization/{dataset}/C3/seed12345/natural_maps.npz',allow_pickle=False);return z['ids'].astype(str),z['H_matched'],z['IMG_QUERY'],n['gt_masks']
def labels_for(scale,dataset,ids):
 six=pd.read_csv(XSA/'performance/six_readout_per_sample.csv');s=six[(six.scale.eq(scale))&(six.dataset.eq(dataset))&(six.model.eq('mean'))&six.readout.isin(['AUD','OGL'])].pivot(index='sample_id',columns='readout',values='iou').reindex(ids)
 out=pd.DataFrame({'sample_id':ids,'aud_iou':s.AUD.to_numpy(),'ogl_iou':s.OGL.to_numpy()});out['ogl_minus_aud_iou']=out.ogl_iou-out.aud_iou
 if scale=='144k' and dataset=='vggss':
  r=pd.read_csv(RES/'per_sample/vggss_aud_vs_ogl_per_sample.csv');out=out.merge(r[['sample_id','group','auto_taxonomy','delta_iou','aud_precision','aud_coverage','aud_area']],on='sample_id',how='left',validate='one_to_one');out['groupA']=out.group.eq('A_ogl_help');out['aud_better']=out.group.eq('C_aud_better');out['ogl_help']=out.group.eq('A_ogl_help')
 else:
  out['group']=np.where(out.ogl_minus_aud_iou>.10,'A_ogl_help',np.where(out.ogl_minus_aud_iou<-.10,'C_aud_better','other'));out['groupA']=False;out['aud_better']=out.group.eq('C_aud_better');out['ogl_help']=out.group.eq('A_ogl_help')
 return out
def process(scale,dataset):
 ids,audraw,imgraw,gt=sources(scale,dataset);prior=np.load(EXP/f'stage_specific_audit/degradation_path/{dataset}/object_prior.npz',allow_pickle=False);fixed=np.load(E80/f'results/{dataset}_144k_sample_indices.npz',allow_pickle=False);assert np.array_equal(ids,fixed['ids'].astype(str)) and np.array_equal(ids,prior['ids'].astype(str));lab=labels_for(scale,dataset,ids);device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');rows=[];signalrows=[];readrows=[]
 for st in range(0,len(ids),32):
  en=min(st+32,len(ids));aud=resize_norm(audraw[st:en],device);qry=resize_norm(imgraw[st:en],device);pr=resize_norm(prior['maps'][st:en],device);iqr=norm(torch.from_numpy(.6*aud+.4*qry)).numpy();ogl=norm(torch.from_numpy(.6*aud+.4*pr)).numpy()
  for j,i in enumerate(range(st,en)):
   gtb=gt[i]>=.5;near=np.zeros((224,224),bool);ni=fixed['indices'][i,2,:int(fixed['counts'][i,2])];near.ravel()[ni]=True;near &= (~gtb);bg=(~gtb) & (~near)
   A,Q,I,O=aud[j],qry[j],iqr[j],ogl[j];ex=np.maximum(A-Q,0);qx=np.maximum(Q-A,0);pe=ex>0;ap=A>=.6;op=O>=.6;remove=ap & (~op);rfp=remove & (~gtb);rtp=remove & gtb;retainfp=ap & op & (~gtb);retaintp=ap & op & gtb
   total=ex.sum();parts=[ex[gtb].sum(),ex[near].sum(),ex[bg].sum()]
   row={'scale':scale,'dataset':dataset,'sample_id':ids[i],'sample_index':i,'positive_excess_total_mass':float(total),'positive_excess_mean':float(ex.mean()),'positive_excess_max':float(ex.max()),'positive_excess_area_gt0':float(pe.mean()),'positive_excess_area_ge_0p1':float((ex>=EXCESS_AREA_THRESHOLD).mean()),'positive_excess_aud_mass_ratio':float(total/max(A.sum(),EPS)),'query_excess_total_mass':float(qx.sum()),'query_excess_mean':float(qx.mean()),'query_excess_max':float(qx.max()),'query_excess_area_gt0':float((qx>0).mean()),'excess_gt_mass':float(parts[0]),'excess_nearfp_mass':float(parts[1]),'excess_bg_mass':float(parts[2]),'excess_gt_ratio':float(parts[0]/max(total,EPS)),'excess_nearfp_ratio':float(parts[1]/max(total,EPS)),'excess_bg_ratio':float(parts[2]/max(total,EPS)),'excess_fp_ratio':float((parts[1]+parts[2])/max(total,EPS)),'removefp_count':int(rfp.sum()),'removetp_count':int(rtp.sum()),'addfp_count':int((op & (~ap) & (~gtb)).sum()),'addtp_count':int((op & (~ap) & gtb).sum()),'overlap_removefp_excess':float((rfp & pe).sum()/max(rfp.sum(),1)),'excess_precision_removefp':float((rfp & pe).sum()/max(pe.sum(),1)),'overlap_removetp_excess':float((rtp & pe).sum()/max(rtp.sum(),1)),'excess_precision_removetp':float((rtp & pe).sum()/max(pe.sum(),1)),'excess_mean_removefp':float(ex[rfp].mean()) if rfp.any() else np.nan,'excess_mean_removetp':float(ex[rtp].mean()) if rtp.any() else np.nan,'excess_mean_retainedfp':float(ex[retainfp].mean()) if retainfp.any() else np.nan,'excess_mean_retaintp':float(ex[retaintp].mean()) if retaintp.any() else np.nan}
   need_pixel=(scale=='144k' and dataset=='flickr') or (scale=='144k' and dataset=='vggss' and bool(lab.iloc[i].groupA or lab.iloc[i].aud_better))
   if need_pixel:
    for name,score in [('positive_excess',ex),('aud_minus_iqr',np.maximum(A-I,0)),('aud_times_one_minus_query',A*(1-Q)),('aud_query_ratio',A/(Q+1e-3))]:
     au1,ap1=aucap(score,rfp,rtp);au2,ap2=aucap(score,rfp,retainfp);signalrows.append({'scale':scale,'dataset':dataset,'sample_id':ids[i],'sample_index':i,'signal':name,'auroc_removefp_vs_removetp':au1,'ap_removefp_vs_removetp':ap1,'auroc_removefp_vs_retainedfp':au2,'ap_removefp_vs_retainedfp':ap2})
   if need_pixel:
    for name,score in [('AUD',A),('IMG_QUERY',Q),('IQR',I),('OGL',O)]:
     pred=score>=.6;inter=(pred*gt[i]).sum();gi=fixed['indices'][i,0,:int(fixed['counts'][i,0])];nv=score.ravel()[ni];gv=score.ravel()[gi];readrows.append({'scale':scale,'dataset':dataset,'sample_id':ids[i],'sample_index':i,'readout':name,'precision':float(inter/max(pred.sum(),EPS)),'coverage':float(inter/max(gt[i].sum(),EPS)),'activated_area':float(pred.mean()),'gt_nearfp_gap':float(gv.mean()-nv.mean()) if len(ni) else np.nan,'gt_response':float(gv.mean()),'nearfp_response':float(nv.mean()) if len(ni) else np.nan})
   rows.append(row)
 df=pd.DataFrame(rows).merge(lab,on=['sample_id'],how='left',validate='one_to_one');sig=pd.DataFrame(signalrows,columns=['scale','dataset','sample_id','sample_index','signal','auroc_removefp_vs_removetp','ap_removefp_vs_removetp','auroc_removefp_vs_retainedfp','ap_removefp_vs_retainedfp']).merge(lab[['sample_id','ogl_minus_aud_iou','groupA','ogl_help','aud_better']],on='sample_id');reads=pd.DataFrame(readrows).merge(lab[['sample_id','groupA','ogl_help','aud_better','ogl_minus_aud_iou']],on='sample_id')
 return df,sig,reads
def main():
 for d in ['alignment','per_sample','spatial_decomposition','ogl_overlap','groupA','safety','iqr','signals','figures','summary']:(HERE/d).mkdir(parents=True,exist_ok=True)
 results={}
 for scale,dataset in [('144k','vggss'),('144k','flickr'),('10k','vggss'),('10k','flickr')]:
  df,sig,reads=process(scale,dataset);df.to_csv(HERE/f'per_sample/{dataset}_{scale}_positive_excess.csv',index=False);sig.to_csv(HERE/f'signals/{dataset}_{scale}_per_sample_signals.csv',index=False);reads.to_csv(HERE/f'iqr/{dataset}_{scale}_readout_per_sample.csv',index=False);results[(scale,dataset)]=(df,sig,reads)
 # Main VGG144 comparisons and frozen safety groups.
 v,s,r=results[('144k','vggss')];a=v.groupA;metrics=['positive_excess_total_mass','positive_excess_aud_mass_ratio','excess_gt_mass','excess_nearfp_mass','excess_bg_mass','excess_gt_ratio','excess_nearfp_ratio','excess_bg_ratio','excess_fp_ratio','overlap_removefp_excess','excess_precision_removefp','overlap_removetp_excess','excess_precision_removetp','excess_mean_removefp','excess_mean_removetp']
 pd.DataFrame(flat('fixed Group-A vs non-Group-A',{m:compare(v.loc[a,m],v.loc[~a,m],'a'+m) for m in metrics})).to_csv(HERE/'groupA/groupA_excess_summary.csv',index=False)
 pd.DataFrame(flat('OGL-help vs AUD-better',{m:compare(v.loc[v.ogl_help,m],v.loc[v.aud_better,m],'safe'+m) for m in metrics})).to_csv(HERE/'safety/aud_better_vs_ogl_help.csv',index=False)
 # spatial rows in a long universal table
 long=[]
 for (scale,dataset),(df,_,_) in results.items():
  for m in ['excess_gt_mass','excess_nearfp_mass','excess_bg_mass','excess_gt_ratio','excess_nearfp_ratio','excess_bg_ratio','excess_fp_ratio']:
   long.append({'scale':scale,'dataset':dataset,'metric':m,**desc(df[m],f'{scale}{dataset}{m}')})
 pd.DataFrame(long).to_csv(HERE/'spatial_decomposition/excess_gt_nearfp_bg.csv',index=False)
 # Signal summaries and correlations.
 ss=[]
 for (scale,dataset),(df,sig,_) in results.items():
  for signal,x in sig.groupby('signal'):
   row={'scale':scale,'dataset':dataset,'signal':signal}
   for metric in ['auroc_removefp_vs_removetp','ap_removefp_vs_removetp','auroc_removefp_vs_retainedfp','ap_removefp_vs_retainedfp']:row[metric]=float(x[metric].mean(skipna=True))
   merged=x.merge(df[['sample_id','positive_excess_total_mass','excess_fp_ratio']],on='sample_id');row['rho_signal_auroc_vs_ogl_gain']=float(spearmanr(merged.auroc_removefp_vs_removetp,merged.ogl_minus_aud_iou,nan_policy='omit').statistic);ss.append(row)
 pd.DataFrame(ss).to_csv(HERE/'signals/positive_side_signal_summary.csv',index=False)
 # Group-A excess correlation with gain and precision gain (OGL precision-AUD precision uses previous residual).
 rv=pd.read_csv(RES/'per_sample/vggss_aud_vs_ogl_per_sample.csv')[['sample_id','ogl_precision','aud_precision']];vg=v.merge(rv,on='sample_id');ga=vg[vg.groupA];corr=[]
 for m in ['positive_excess_total_mass','excess_fp_ratio','overlap_removefp_excess','excess_precision_removefp']:
  for y in ['ogl_minus_aud_iou']:
   corr.append({'subset':'Group-A','metric':m,'outcome':y,'spearman_rho':float(spearmanr(ga[m],ga[y],nan_policy='omit').statistic)})
  corr.append({'subset':'Group-A','metric':m,'outcome':'ogl_minus_aud_precision','spearman_rho':float(spearmanr(ga[m],ga.ogl_precision-ga.aud_precision,nan_policy='omit').statistic)})
 pd.DataFrame(corr).to_csv(HERE/'ogl_overlap/excess_correlations.csv',index=False)
 # Transitions/readouts on group A, all reported on same aligned grid.
 groupreads=r[r.groupA].groupby('readout').mean(numeric_only=True).reset_index();groupreads.to_csv(HERE/'iqr/aud_img_iqr_ogl_comparison.csv',index=False)
 v.to_csv(HERE/'ogl_overlap/excess_vs_removefp.csv',index=False)
 sig[sig.groupA].groupby('signal').mean(numeric_only=True).reset_index().to_csv(HERE/'ogl_overlap/pixel_prediction_metrics.csv',index=False)
 (HERE/'summary/analysis_bundle.json').write_text(json.dumps({'read_only':True,'alignment':'224 bicubic then per-map minmax; IQR=minmax(.6 AUD+.4 IMG_QUERY); threshold=.6; PositiveExcess=max(A-Q,0)','excess_area_threshold':EXCESS_AREA_THRESHOLD},indent=2))
 print('completed positive excess audit')
if __name__=='__main__':main()
