#!/usr/bin/env python3
"""Frozen Hard-CRAM vs Mean-CRAM final readout and mechanism audit.

This evaluator never trains or writes a checkpoint.  It reuses the exact 8.1
wrong-audio indices, 8.0 GT/NearFP region samples, 8.7 normalization and
bootstrap routines.  The Hard teacher's own audio queries are recomputed from
the cached test spectrograms; using the Mean teacher's saved queries would be
incorrect.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, importlib.util, json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE=Path(__file__).resolve().parent; HARD=HERE.parent; MEAN=HARD.parent
ROOT=MEAN.parents[1]; RECIPE=ROOT/'chuagnxindian/1mufasaslot/1.3G-multigeom_equivariant_l3_refine'
EXP86=ROOT/'chuagnxindian/8.6_loss_to_decision_causal_ablation'
MECH=MEAN/'mechanism_audit'; EXP81=ROOT/'chuagnxindian/8.1_same_image_audio_counterfactual_audit'
EXP80=ROOT/'chuagnxindian/8.0_audio_only_residual_audit'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module; spec.loader.exec_module(module); return module

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def attention(q,k,sharpening):
    logits=torch.einsum('bksd,bpd->bksp',q,k)*(q.shape[-1]**-.5)*sharpening
    value=logits.softmax(2)+1e-8
    return value/value.sum(3,keepdim=True)

def build(dataset,device):
    sys.path.insert(0,str(RECIPE)); import common as C
    registry=copy.deepcopy(C.EXPERIMENTS[f'{dataset}_144k'])
    teacher=HARD/f'stage1/{dataset}/seed12345/selected_best.pth'
    student=HARD/f'stage2/{dataset}/seed12345/{dataset}_best.pth'
    registry['base_checkpoint']=str(teacher.resolve())
    cfg=C.load_base_config(registry); cfg.workers=8
    C.setup_seed(12345); model,loaded=C.build_model(cfg,registry,device)
    if loaded.resolve()!=teacher.resolve(): raise RuntimeError(f'wrong teacher: {loaded}')
    ck=torch.load(student,map_location='cpu',weights_only=False)
    model.student.proj3_spatial.load_state_dict(ck['proj3_spatial_state_dict'],strict=True)
    model.student.adapter.load_state_dict(ck['topdown_adapter_state_dict'],strict=True)
    model.eval(); return C,model,teacher,student,ck

@torch.inference_mode()
def extract(dataset,gpu,out):
    device=torch.device(f'cuda:{gpu}'); C,model,teacher,student,ck=build(dataset,device)
    cache=load(f'hard_eval_cache_{dataset}',EXP86/'common.py'); data=cache.test_dataset(dataset)
    ids=np.asarray([Path(x).stem for x in data.image_files]).astype(str); n=len(ids)
    with np.load(EXP81/f'results/{dataset}_144k_counterfactual_indices.npz') as z:
        if not np.array_equal(ids,z['anchor_ids'].astype(str)): raise RuntimeError('8.1 ids mismatch')
        wrong=z['set1_indices'][:,:8].astype(np.int64)
    bs=128 if dataset=='vggss' else 32
    loader=DataLoader(data,batch_size=bs,shuffle=False,num_workers=8,pin_memory=True,persistent_workers=True)
    queries=np.empty((n,2,512),np.float32); pos=0
    for _image,audio,_gt,names,_ in loader:
        audio=audio.to(device,non_blocking=True).float(); q=model._extract_audio_query(audio)
        queries[pos:pos+len(names)]=q.cpu().numpy();pos+=len(names)
    if pos!=n: raise RuntimeError('query extraction length')
    matched=np.empty((n,196),np.float32); wrong_maps=np.empty((n,8,196),np.float32); image_maps=np.empty((n,49),np.float32)
    replay=0.;pos=0
    for image,audio,_gt,names,_ in loader:
        b=len(names); ix=np.arange(pos,pos+b); image=image.to(device,non_blocking=True).float()
        q=torch.from_numpy(queries[ix]).to(device); view=model._extract_visual_teacher(image,q); fine=model._fine_from_teacher_features(view,q)
        branch=model.teacher.slot_attn.visual_branches[-1]; tokens=model._to_tokens(fine['F34'])
        keys=branch.img_to_k(branch.img_norm_input(tokens)); hm=attention(q[:,None],keys,model.teacher.infer_sharpening)[:,0,0]
        replay=max(replay,float((hm-fine['AUD_FINE'][:,0].flatten(1)).abs().max()))
        matched[ix]=hm.cpu().numpy(); qw=torch.from_numpy(queries[wrong[ix]]).to(device)
        wrong_maps[ix]=attention(qw,keys,model.teacher.infer_sharpening)[:,:,0].cpu().numpy()
        # Exact defined IMG_QUERY branch of the original six-readout evaluator.
        img,_aud=model.teacher.forward_eval(image,audio.to(device,non_blocking=True).float())
        image_maps[ix]=img[:,0].flatten(1).cpu().numpy();pos+=b
        if pos%512==0 or pos==n: print(dataset,'extract',pos,n,flush=True)
    np.savez_compressed(out/'hard_maps.npz',ids=ids,H_matched=matched,H_wrong=wrong_maps,wrong_indices=wrong,IMG_QUERY=image_maps)
    (out/'extraction.json').write_text(json.dumps({'dataset':dataset,'teacher':str(teacher),'teacher_sha256':sha(teacher),'student':str(student),'student_sha256':sha(student),'student_epoch':int(ck['epoch']),'ids':n,'offline_replay_max_abs_error':replay},indent=2)+'\n')
    model.close()

def normalize(x):
    lo=x.min(axis=(-2,-1),keepdims=True); hi=x.max(axis=(-2,-1),keepdims=True)
    return (x-lo)/np.maximum(hi-lo,1e-12)

def summarize(dataset,gpu,out):
    A=load(f'hard_metrics_{dataset}',MECH/'scripts/analyze.py'); A.MODELS=('original_1.3g_final','mean_cram','hard_cram'); A.DISPLAY={'mean_cram':'Mean-CRAM','hard_cram':'Hard-CRAM'}
    device=torch.device(f'cuda:{gpu}')
    with np.load(HARD/f'mechanism/{dataset}/hard_maps.npz') as z:
        ids=z['ids'].astype(str); hard={'ids':ids,'matched':z['H_matched'],'wrong':z['H_wrong'],'wrong_indices':z['wrong_indices']}; hard_img=z['IMG_QUERY']
    with np.load(MECH/f'results/{dataset}/full/cram_c3_stage2_maps.npz') as z:
        if not np.array_equal(ids,z['ids'].astype(str)): raise RuntimeError('mean-map ids mismatch')
        mean={'ids':ids,'matched':z['H_matched'],'wrong':z['H_wrong'],'wrong_indices':z['wrong_indices']}
    with np.load(MECH/f'results/{dataset}/full/original_1.3g_final_maps.npz') as z:
        if not np.array_equal(ids,z['ids'].astype(str)): raise RuntimeError('original-map ids mismatch')
        original={'ids':ids,'matched':z['H_matched'],'wrong':z['H_wrong'],'wrong_indices':z['wrong_indices']}
    with np.load(MEAN/f'natural_localization/{dataset}/C3/seed12345/natural_maps.npz') as z:
        if not np.array_equal(ids,z['ids'].astype(str)): raise RuntimeError('natural ids mismatch')
        gt=z['gt_masks'].astype(np.float32); mean_img=z['image_native'].reshape(len(ids),49)
    with np.load(EXP80/f'results/{dataset}_144k_sample_indices.npz') as z: fixed={k:z[k].copy() for k in z.files}
    if not np.array_equal(ids,fixed['ids'].astype(str)): raise RuntimeError('region ids mismatch')
    maps={'original_1.3g_final':original,'mean_cram':mean,'hard_cram':hard}
    cf=pd.concat([A.counterfactual_metrics(dataset,k,v) for k,v in maps.items()],ignore_index=True)
    natural,normed=A.natural_and_response_metrics(dataset,maps,gt,fixed,device)
    # These paired changes are Hard minus Mean, not the legacy comparison to 1.3G.
    mean_norm=normed['mean_cram']; hard_norm=normed['hard_cram']; gb=gt>=.5; mp=mean_norm>=.6; hp=hard_norm>=.6
    add=hp&~mp; rem=mp&~hp
    hard_rows=natural.model.eq('hard_cram')
    natural.loc[hard_rows,'RemoveFP']=(rem&~gb).sum((1,2)); natural.loc[hard_rows,'RemoveTP']=(rem&gb).sum((1,2)); natural.loc[hard_rows,'AddTP']=(add&gb).sum((1,2)); natural.loc[hard_rows,'AddFP']=(add&~gb).sum((1,2))
    transition=pd.DataFrame({'dataset':dataset,'sample_id':ids,'sample_index':np.arange(len(ids)),
        'RemoveFP':(rem&~gb).sum((1,2)),'RemoveTP':(rem&gb).sum((1,2)),
        'AddTP':(add&gb).sum((1,2)),'AddFP':(add&~gb).sum((1,2))})
    # Exact six map definitions established in evaluate_full.py.
    prior_path=MEAN/f'stage_specific_audit/degradation_path/{dataset}/object_prior.npz'
    with np.load(prior_path) as z:
        if not np.array_equal(ids,z['ids'].astype(str)): raise RuntimeError('prior ids mismatch')
        prior=z['maps']
    six=[]
    for name,img in [('mean_cram',mean_img),('hard_cram',hard_img)]:
        _raw,im=A.resize_and_normalize(img,device); _raw,obj=A.resize_and_normalize(prior,device)
        aud=normed[name]
        readouts={'AUD':aud,'IMG_QUERY':im,'IQR':normalize(.6*aud+.4*im),'OBJ_PRIOR':obj,'OGL':normalize(.6*aud+.4*obj),'EXTRA_IQR_OGL':normalize(.6*aud+.2*im+.2*obj)}
        for key,value in readouts.items():
            for i in range(len(ids)): six.append({'dataset':dataset,'model':name,'sample_id':ids[i],'sample_index':i,'readout':key,'iou':A.official_iou(value[i]>=.6,gt[i])})
    six=pd.DataFrame(six)
    # Mean must reproduce its existing formal AUD exactly before Hard is reported.
    existing=pd.read_csv(MEAN/f'stage2/C3/{dataset}/seed12345/best_per_sample.csv',dtype={'sample_id':str})
    # The older Flickr formal file predates an explicit index column; its
    # row-order is the verified evaluation order and is retained verbatim.
    if 'sample_index' not in existing: existing['sample_index']=np.arange(len(existing))
    existing=existing.sort_values('sample_index')
    replay=six[(six.model=='mean_cram')&(six.readout=='AUD')].sort_values('sample_index')
    if not np.array_equal(replay.sample_id.to_numpy(),existing.sample_id.to_numpy()): raise RuntimeError('formal ids mismatch')
    err=float(np.max(np.abs(replay.iou.to_numpy()-existing.AUD_iou.to_numpy())))
    if err>1e-6: raise RuntimeError(f'mean AUD formal replay error {err}')
    metric_cols=['gt_response','nearfp_response','gt_nearfp_gap','gt_vs_nearfp_auroc','coverage','precision','pred_area_ratio','RemoveFP','RemoveTP','AddTP','AddFP','iou']
    paired_metric_cols=[x for x in metric_cols if x not in ('RemoveFP','RemoveTP','AddTP','AddFP')]
    summary=[]; paired=[]
    for name in ('mean_cram','hard_cram'):
        x=natural[natural.model.eq(name)].sort_values('sample_index')
        c=cf[cf.model.eq(name)].sort_values('sample_index')
        row={'dataset':dataset,'model':name}
        for key in metric_cols: row[key]=float(np.nanmean(x[key]))
        row['audio_swap_spearman']=float(c.audio_swap_spearman.mean());row['audio_swap_mae']=float(c.audio_map_mae.mean())
        summary.append(row)
    base=natural[natural.model.eq('mean_cram')].sort_values('sample_index'); candidate=natural[natural.model.eq('hard_cram')].sort_values('sample_index')
    for key in paired_metric_cols+['audio_swap_spearman','audio_map_mae']:
        left=(cf[cf.model.eq('hard_cram')].sort_values('sample_index')[key].to_numpy()-cf[cf.model.eq('mean_cram')].sort_values('sample_index')[key].to_numpy()) if key.startswith('audio_') else candidate[key].to_numpy()-base[key].to_numpy()
        paired.append({'dataset':dataset,'comparison':'Hard-minus-Mean','metric':key,**A.stats(left,dataset,'hard_minus_mean',key)})
    # The frozen VGG residual Group A is defined only once from Mean AUD/OGL.
    groups=np.full(len(ids),'all',dtype=object)
    if dataset=='vggss':
        old=pd.read_csv(MEAN/'residual_ogl_gap_audit/per_sample/vggss_aud_vs_ogl_per_sample.csv',dtype={'sample_id':str}).set_index('sample_id').loc[ids]
        groups=old['group'].to_numpy()
    group=[]
    for label in np.unique(groups):
        sel=groups==label
        for key in paired_metric_cols:
            delta=(candidate[key].to_numpy()-base[key].to_numpy())[sel]
            group.append({'dataset':dataset,'group':str(label),'metric':key,'n':int(sel.sum()),**A.stats(delta,dataset,str(label),key)})
        for key in ('RemoveFP','RemoveTP','AddTP','AddFP'):
            group.append({'dataset':dataset,'group':str(label),'metric':f'Hard_vs_Mean_{key}','n':int(sel.sum()),**A.stats(transition.loc[sel,key].to_numpy(),dataset,str(label),key)})
    transition_summary=[]
    for key in ('RemoveFP','RemoveTP','AddTP','AddFP'):
        transition_summary.append({'dataset':dataset,'comparison':'Hard transition relative to Mean','metric':key,**A.stats(transition[key].to_numpy(),dataset,'all',key)})
    performance=[]
    for (name,key),x in six.groupby(['model','readout'],sort=False):
        v=x.sort_values('sample_index').iou.to_numpy(); thresholds=np.arange(21)*.05
        performance.append({'dataset':dataset,'model':name,'readout':key,'cIoU':float((v>=.5).mean()),'AUC':float(np.trapezoid([(v>=t).mean() for t in thresholds],thresholds))})
    natural.to_csv(out/'object_per_sample.csv',index=False); cf.to_csv(out/'audio_swap_per_sample.csv',index=False); six.to_csv(out/'six_readout_per_sample.csv',index=False);transition.to_csv(out/'hard_vs_mean_transition_per_sample.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'mechanism_summary.csv',index=False);pd.DataFrame(paired).to_csv(out/'paired_hard_minus_mean.csv',index=False);pd.DataFrame(group).to_csv(out/'group_deltas.csv',index=False);pd.DataFrame(transition_summary).to_csv(out/'hard_vs_mean_transition_summary.csv',index=False);pd.DataFrame(performance).to_csv(out/'six_readout_summary.csv',index=False)
    (out/'validation.json').write_text(json.dumps({'mean_aud_iou_max_abs_error_vs_formal':err,'ids_verified':True,'group_A_definition':'frozen prior residual audit' if dataset=='vggss' else 'not applicable'},indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('vggss','flickr'),required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--analyze-only',action='store_true');a=p.parse_args()
    torch.cuda.set_device(a.gpu); out=HERE/a.dataset; out.mkdir(parents=True,exist_ok=True)
    if not a.analyze_only:
        if (out/'hard_maps.npz').exists(): raise RuntimeError('refusing to overwrite hard maps')
        extract(a.dataset,a.gpu,out)
    summarize(a.dataset,a.gpu,out)

if __name__=='__main__': main()
