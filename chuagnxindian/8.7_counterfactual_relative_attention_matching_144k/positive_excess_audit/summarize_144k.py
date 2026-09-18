#!/usr/bin/env python3
"""Read-only postprocessing for the completed 144k positive-excess replay."""
from pathlib import Path
import json, hashlib
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

R=Path(__file__).resolve().parent; N=2000
def rng(k): return np.random.default_rng(int.from_bytes(hashlib.sha256(k.encode()).digest()[:8],'little'))
def summary(x):
 x=np.asarray(x,float); x=x[np.isfinite(x)]; g=rng(str(x.size)+str(x.mean())); b=np.array([x[g.integers(len(x),size=len(x))].mean() for _ in range(N)])
 return dict(n=len(x),mean=x.mean(),median=np.median(x),p25=np.quantile(x,.25),p75=np.quantile(x,.75),ci_low=np.quantile(b,.025),ci_high=np.quantile(b,.975))
def comp(a,b,name):
 a=np.asarray(a,float); b=np.asarray(b,float); a=a[np.isfinite(a)];b=b[np.isfinite(b)];g=rng(name); z=np.array([a[g.integers(len(a),size=len(a))].mean()-b[g.integers(len(b),size=len(b))].mean() for _ in range(N)]); u,p=mannwhitneyu(a,b)
 return dict(metric=name,**{'left_'+k:v for k,v in summary(a).items()},**{'right_'+k:v for k,v in summary(b).items()},mean_diff=a.mean()-b.mean(),diff_ci_low=np.quantile(z,.025),diff_ci_high=np.quantile(z,.975),mw_p=p,rank_biserial=2*u/(len(a)*len(b))-1)
v=pd.read_csv(R/'per_sample/vggss_144k_positive_excess.csv'); f=pd.read_csv(R/'per_sample/flickr_144k_positive_excess.csv')
metrics=['positive_excess_total_mass','positive_excess_aud_mass_ratio','excess_gt_ratio','excess_nearfp_ratio','excess_bg_ratio','excess_fp_ratio','overlap_removefp_excess','overlap_removetp_excess','excess_precision_removefp','excess_precision_removetp']
out=[comp(v[v.groupA][m],v[~v.groupA][m],'GroupA_minus_nonA:'+m) for m in metrics]
pd.DataFrame(out).to_csv(R/'groupA/groupA_excess_summary.csv',index=False)
out=[comp(v[v.ogl_help][m],v[v.aud_better][m],'OGLhelp_minus_AUDbetter:'+m) for m in metrics]
pd.DataFrame(out).to_csv(R/'safety/aud_better_vs_ogl_help.csv',index=False)
sp=[]
for name,d in [('vggss_144k',v),('flickr_144k',f)]:
 for m in metrics: sp.append(dict(dataset=name,metric=m,**summary(d[m])))
pd.DataFrame(sp).to_csv(R/'spatial_decomposition/excess_gt_nearfp_bg.csv',index=False)
c=[]
for subset,d in [('VGG_GroupA',v[v.groupA]),('VGG_all',v),('Flickr_all',f)]:
 for x in ['positive_excess_total_mass','positive_excess_aud_mass_ratio','excess_fp_ratio','overlap_removefp_excess']:
  for y in ['ogl_minus_aud_iou']:
   q=spearmanr(d[x],d[y],nan_policy='omit');c.append(dict(subset=subset,signal=x,outcome=y,rho=q.statistic,p=q.pvalue))
pd.DataFrame(c).to_csv(R/'ogl_overlap/excess_correlations.csv',index=False)
read=[]
for ds,d in [('vggss',v),('flickr',f)]:
 for group,x in [('all',d),('groupA',d[d.groupA]),('ogl_help',d[d.ogl_help]),('aud_better',d[d.aud_better])]:
  if len(x): read.append(dict(dataset=ds,subset=group,n=len(x),**{m:x[m].mean() for m in metrics}))
pd.DataFrame(read).to_csv(R/'safety/vgg_flickr_safety_summary.csv',index=False)
msg={'decision':'S2','reason':'Group-A positive excess is FP/background enriched and intersects OGL RemoveFP, but it does not distinguish RemoveFP from RemoveTP (pixel AUROC ~0.49), while AUD-better samples have substantially more GT excess and more RemoveTP overlap. Direct suppression is unsafe.','alignment':'Formal evaluation grid: each native map independently bicubic-resized to 224x224 then per-map min-max normalized. IQR=minmax(0.6*AUD+0.4*IMG_QUERY); OGL=minmax(0.6*AUD+0.4*OBJ_PRIOR). PositiveExcess=max(AUD-IMG_QUERY,0). Threshold .6 only for existing AUD-to-OGL transition masks; excess itself stays continuous.'}
(R/'summary/next_action_plan.json').write_text(json.dumps(msg,indent=2))
