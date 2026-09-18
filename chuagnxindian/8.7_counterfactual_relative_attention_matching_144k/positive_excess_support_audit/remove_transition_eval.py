from pathlib import Path
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
R=Path(__file__).resolve().parent;E=R.parent
def n(x):return (x-x.amin((-2,-1),True))/(x.amax((-2,-1),True)-x.amin((-2,-1),True)+1e-8)
ds='vggss';z=np.load(R/'candidate_scores_vggss_5158.npz');m=np.load(E/'mechanism_audit/results/vggss/full/cram_c3_stage2_maps.npz');q=np.load(E/'natural_localization/vggss/C3/seed12345/natural_maps.npz')['image_native'];p=np.load(E/'stage_specific_audit/degradation_path/vggss/object_prior.npz')['maps'].reshape(-1,7,7);g=pd.read_csv(E/'positive_excess_audit/per_sample/vggss_144k_positive_excess.csv').groupA.to_numpy(bool)
rows=[]
for j,name in enumerate(z['names']):
 vals=[];labs=[]
 for st in range(0,len(g),64):
  ix=np.arange(st,min(st+64,len(g)));ix=ix[g[ix]]
  if not len(ix):continue
  s=F.interpolate(torch.from_numpy(z['scores'][ix,j].astype('float32'))[:,None],(224,224),mode='bicubic',align_corners=False)[:,0].numpy();a=n(F.interpolate(torch.from_numpy(m['H_matched'][ix].reshape(-1,14,14))[:,None],(224,224),mode='bicubic',align_corners=False)[:,0]).numpy();qq=n(F.interpolate(torch.from_numpy(q[ix])[:,None],(224,224),mode='bicubic',align_corners=False)[:,0]).numpy();o=n(F.interpolate(torch.from_numpy(p[ix])[:,None],(224,224),mode='bicubic',align_corners=False)[:,0]).numpy();ogl=n(torch.from_numpy(.6*a+.4*o)).numpy();ex=(a-qq)>=.1;rm=(a>=.6)&(ogl<.6);gt=np.load(E/'natural_localization/vggss/C3/seed12345/natural_maps.npz')['gt_masks'][ix]>=.5;fp=rm & (~gt) & ex;tp=rm & gt & ex;vals.append(np.r_[s[fp],s[tp]]);labs.append(np.r_[np.zeros(fp.sum()),np.ones(tp.sum())])
 y=np.concatenate(labs);v=np.concatenate(vals);rows.append({'signal':name,'auroc_removetp_vs_removefp':roc_auc_score(y,v),'ap_removetp':average_precision_score(y,v),'n':len(y)})
pd.DataFrame(rows).to_csv(R/'ogl_alignment/vgg_groupA_removefp_removetp.csv',index=False)
