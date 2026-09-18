#!/usr/bin/env python3
"""Fast descriptive native-grid readout/visual-key completion (read-only)."""
from pathlib import Path
import json,numpy as np,pandas as pd
from run_audit import HERE,EXP,E80,EPS,comp,flatten
def norm(x):
 lo=x.min(axis=(1,2),keepdims=True);hi=x.max(axis=(1,2),keepdims=True);return (x-lo)/np.maximum(hi-lo,EPS)
def metric(score,gt,near):
 p=score>=.6;gb=gt>=.5;inter=(p*gt).sum((1,2));
 return {'gt_response':(score*gt).sum((1,2))/np.maximum(gt.sum((1,2)),EPS),'nearfp_response':(score*near).sum((1,2))/np.maximum(near.sum((1,2)),EPS),'gt_nearfp_gap':None,'precision':inter/np.maximum(p.sum((1,2)),EPS),'coverage':inter/np.maximum(gt.sum((1,2)),EPS),'activated_area':p.mean((1,2))}
def main():
 audit=pd.read_csv(HERE/'positive_map/per_sample_positive_metrics.csv');a=audit.group.eq('A_ogl_help');ids_a=audit.loc[a,'sample_id'].to_numpy()
 maps=np.load(EXP/'mechanism_audit/results/vggss/full/cram_c3_stage2_maps.npz',allow_pickle=False);nat=np.load(EXP/'natural_localization/vggss/C3/seed12345/natural_maps.npz',allow_pickle=False);fixed=np.load(E80/'results/vggss_144k_sample_indices.npz',allow_pickle=False);ids=maps['ids'].astype(str);lookup={s:i for i,s in enumerate(ids)};ii=np.array([lookup[s] for s in ids_a])
 aud=norm(maps['H_matched'][ii].reshape(-1,14,14));img7=norm(nat['image_native'][ii]);img=np.repeat(np.repeat(img7,2,1),2,2);iqr=norm(.6*aud+.4*img)
 gt=nat['gt_masks'][ii].reshape(-1,14,16,14,16).mean((2,4));near=np.zeros((len(ii),224,224),float)
 for j,i in enumerate(ii):
  z=fixed['indices'][i,2,:int(fixed['counts'][i,2])];near[j].reshape(-1)[z]=1
 near=near.reshape(-1,14,16,14,16).mean((2,4));rows=[]
 for name,score in [('AUD',aud),('IMG_QUERY',img),('IQR',iqr)]:
  z=metric(score,gt,near);gap=z['gt_response']-z['nearfp_response']
  for j,s in enumerate(ids_a):rows.append({'sample_id':s,'sample_index':int(ii[j]),'readout':name,'grid':'native_14x14_descriptive','gt_response':z['gt_response'][j],'nearfp_response':z['nearfp_response'][j],'gt_nearfp_gap':gap[j],'precision':z['precision'][j],'coverage':z['coverage'][j],'activated_area':z['activated_area'][j]})
 rr=pd.DataFrame(rows);rr.to_csv(HERE/'readout_comparison/aud_img_iqr_groupA_per_sample.csv',index=False);rr.groupby(['readout','grid']).mean(numeric_only=True).reset_index().to_csv(HERE/'readout_comparison/aud_img_iqr_groupA.csv',index=False)
 keys=np.load(EXP/'natural_localization/vggss/C3/seed12345/visual_keys.npy',mmap_mode='r');gt7=nat['gt_masks'].reshape(-1,7,32,7,32).mean((2,4));near7=np.zeros((len(ids),224,224),float)
 for i in range(len(ids)):
  z=fixed['indices'][i,2,:int(fixed['counts'][i,2])];near7[i].reshape(-1)[z]=1
 near7=near7.reshape(-1,7,32,7,32).mean((2,4));gv=(keys*gt7.reshape(len(ids),49,1)).sum(1)/np.maximum(gt7.reshape(len(ids),49).sum(1,keepdims=True),EPS);nv=(keys*near7.reshape(len(ids),49,1)).sum(1)/np.maximum(near7.reshape(len(ids),49).sum(1,keepdims=True),EPS);cos=(gv*nv).sum(1)/np.maximum(np.linalg.norm(gv,axis=1)*np.linalg.norm(nv,axis=1),EPS)
 sim=pd.DataFrame({'sample_id':ids,'sample_index':np.arange(len(ids)),'gt_nearfp_visual_cosine':cos}).merge(audit[['sample_id','group']],on='sample_id');sim.to_csv(HERE/'visual_similarity/gt_nearfp_feature_similarity.csv',index=False);z=comp(sim.loc[sim.group.eq('A_ogl_help'),'gt_nearfp_visual_cosine'],sim.loc[~sim.group.eq('A_ogl_help'),'gt_nearfp_visual_cosine'],'visual');pd.DataFrame(flatten('fixed Group-A vs non-Group-A C3 visual cosine',{'gt_nearfp_visual_cosine':z})).to_csv(HERE/'visual_similarity/groupA_visual_similarity_summary.csv',index=False)
 bundle=HERE/'summary/analysis_bundle.json';old=json.loads(bundle.read_text()) if bundle.exists() else {};old.update({'phase_B':'completed: full-set exact cached AUD/OGL metrics plus native-grid IMG/IQR readout replay','visual_similarity':z});bundle.write_text(json.dumps(old,indent=2));print('completed')
if __name__=='__main__':main()
