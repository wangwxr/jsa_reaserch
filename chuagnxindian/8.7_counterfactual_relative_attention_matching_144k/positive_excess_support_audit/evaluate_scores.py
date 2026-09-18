from pathlib import Path
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
R=Path(__file__).resolve().parent; E=R.parent
def norm(x): return (x-x.amin((-2,-1),True))/(x.amax((-2,-1),True)-x.amin((-2,-1),True)+1e-8)
def run(ds):
 z=np.load(R/f'candidate_scores_{ds}_{5158 if ds=="vggss" else 250}.npz'); m=np.load(E/f'mechanism_audit/results/{ds}/full/cram_c3_stage2_maps.npz'); n=np.load(E/f'natural_localization/{ds}/C3/seed12345/natural_maps.npz'); p=np.load(E/f'stage_specific_audit/degradation_path/{ds}/object_prior.npz');assert np.array_equal(z['ids'],m['ids'])
 # Retain only native arrays; formal 224 maps are built in chunks below.
 raw_s=z['scores'].astype('float32'); raw_a=m['H_matched'].reshape(-1,14,14); raw_q=n['image_native']; side=int(np.sqrt(p['maps'].shape[1])); raw_o=p['maps'].reshape(-1,side,side); gt=n['gt_masks']>=.5
 rows=[]
 groups=pd.read_csv(E/'positive_excess_audit/per_sample/vggss_144k_positive_excess.csv') if ds=='vggss' else pd.read_csv(E/'positive_excess_audit/per_sample/flickr_144k_positive_excess.csv')
 for j,name in enumerate(z['names']):
  for subset,keep in [('all',np.ones(len(raw_s),bool)),('groupA',groups.groupA.to_numpy(bool)),('aud_better',groups.aud_better.to_numpy(bool))]:
   vals=[];labs=[]
   if not keep.any(): continue
   for st in range(0,len(raw_s),64):
    ix=np.arange(st,min(st+64,len(raw_s))); ix=ix[keep[ix]]
    if not len(ix): continue
    sc=F.interpolate(torch.from_numpy(raw_s[ix,j])[:,None],(224,224),mode='bicubic',align_corners=False)[:,0].numpy(); aa=norm(F.interpolate(torch.from_numpy(raw_a[ix])[:,None],(224,224),mode='bicubic',align_corners=False)[:,0]).numpy(); qq=norm(F.interpolate(torch.from_numpy(raw_q[ix])[:,None],(224,224),mode='bicubic',align_corners=False)[:,0]).numpy(); mask=(aa-qq)>=.1; tp=gt[ix]&mask;fp=(~gt[ix])&mask; vals.append(np.r_[sc[tp],sc[fp]]);labs.append(np.r_[np.ones(tp.sum()),np.zeros(fp.sum())])
   y=np.concatenate(labs);v=np.concatenate(vals)
   if len(np.unique(y))>1: rows.append(dict(dataset=ds,subset=subset,signal=name,auroc_tp_vs_fp=roc_auc_score(y,v),ap_tp=average_precision_score(y,v),n=len(y)))
 pd.DataFrame(rows).to_csv(R/f'ogl_alignment/{ds}_tp_fp_metrics.csv',index=False)
if __name__=='__main__':
 run('vggss');run('flickr')
