"""Frozen full-test degradation audit reusing existing maps, regions and metrics."""
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gradient_conflict'))
from run import T, HERE, EXP, load, save_json
A=load('stage_specific_metrics',EXP/'mechanism_audit/scripts/analyze.py')
X=load('stage_specific_extract',EXP/'mechanism_audit/scripts/extract_stage2_maps.py')

def normalize(x):
    lo=x.min(axis=(-2,-1),keepdims=True);hi=x.max(axis=(-2,-1),keepdims=True)
    return (x-lo)/np.maximum(hi-lo,1e-12)

def source_paths(ds):
    paths={f'short_step{s:03d}':HERE/f'gradient_conflict/{ds}/step{s:03d}.pth' for s in (0,16,64)}
    for lam in ('025','050'):
        for tag,file in [('best',f'{ds}_best.pth'),('latest','latest.pth')]:
            paths[f'lambda{lam}k_{tag}']=EXP/f'stage2_with_cram/checkpoints/lambda{lam}k/{ds}/seed12345/{file}'
    return paths

@torch.inference_mode()
def extract(ds,device,outdir):
    model,config,registry,teacher=T.build_model(ds,device);model.eval()
    paths=source_paths(ds);states={k:torch.load(v,map_location=device,weights_only=False) for k,v in paths.items()}
    natural=np.load(EXP/f'natural_localization/{ds}/C3/seed12345/natural_maps.npz')
    ids=natural['ids'].astype(str);n=len(ids)
    queries=np.load(EXP/f'natural_localization/{ds}/C3/seed12345/audio_queries.npy')
    with np.load(X.EXP81/f'results/{ds}_144k_counterfactual_indices.npz') as z:
        assert np.array_equal(ids,z['anchor_ids'].astype(str)); wrong=z['set1_indices'][:,:8].astype(int)
    cache=T.load_module('stage_specific_cache',T.EXP86/'common.py')
    dataset=cache.test_dataset(ds)
    assert np.array_equal(ids,np.array([Path(v).stem for v in dataset.image_files]))
    loader=DataLoader(dataset,batch_size=64,shuffle=False,num_workers=6,pin_memory=True)
    full=T.load_module('stage_specific_full',T.RECIPE/'evaluate_full.py')
    prior=full.object_prior_model().to(device).eval()
    prior_maps=np.empty((n,49),np.float32)
    maps={k:{'matched':np.empty((n,196),np.float32),'wrong':np.empty((n,8,196),np.float32)} for k in paths}
    pos=0;error=0.;ids_seen=[]
    for image,audio,gt,names,_ in loader:
        image,audio,gt,names=T.original_common.flatten_eval_batch(image,audio,gt,names)
        image=image.to(device).float();batch=len(image);ix=np.arange(pos,pos+batch)
        q=torch.from_numpy(queries[ix]).to(device);qw=torch.from_numpy(queries[wrong[ix]]).to(device)
        features=model._extract_visual_teacher(image,q)
        prior_maps[ix]=prior(image).flatten(1).cpu().numpy()
        for key,ck in states.items():
            model.student.proj3_spatial.load_state_dict(ck['proj3_spatial_state_dict']);model.student.adapter.load_state_dict(ck['topdown_adapter_state_dict'])
            fine=model._fine_from_teacher_features(features,q)
            hm=X.offline_attention(q[:,None],fine['FINE_KEYS'],model.teacher.infer_sharpening)[:,0,0]
            error=max(error,float((hm-fine['AUD_FINE'].flatten(1)).abs().max()))
            maps[key]['matched'][ix]=hm.cpu().numpy()
            maps[key]['wrong'][ix]=X.offline_attention(qw,fine['FINE_KEYS'],model.teacher.infer_sharpening)[:,:,0].cpu().numpy()
        ids_seen.extend(names);pos+=batch
        if pos%512==0: print(ds,'extract',pos,n,flush=True)
    assert np.array_equal(ids,ids_seen) and error<1e-6
    for key,mm in maps.items():
        np.savez_compressed(outdir/f'{key}_maps.npz',ids=ids,H_matched=mm['matched'],H_wrong=mm['wrong'],wrong_indices=wrong)
    np.savez_compressed(outdir/'object_prior.npz',ids=ids,maps=prior_maps)
    save_json(outdir/'extraction_manifest.json',dict(teacher=str(teacher),teacher_sha256=T.sha256(teacher),
        checkpoints={k:dict(path=str(p),sha256=T.sha256(p),epoch=states[k]['epoch']) for k,p in paths.items()},
        replay_max_error=error,all_ids_verified=True,training_used=False,
        original_audio_query_shape=list(queries.shape),infer_sharpening=model.teacher.infer_sharpening))
    model.close()

def analyze(ds,device,outdir):
    with np.load(EXP/f'natural_localization/{ds}/C3/seed12345/natural_maps.npz') as z:
        ids=z['ids'].astype(str);gt=z['gt_masks'];img=z['image_native'].reshape(len(ids),49)
    with np.load(A.EXP80/f'results/{ds}_144k_sample_indices.npz') as z:
        fixed={k:z[k].copy() for k in z.files}
    assert np.array_equal(ids,fixed['ids'].astype(str))
    maps={k:A.load_maps(ds,k) for k in A.MODELS}
    for key in source_paths(ds):
        with np.load(outdir/f'{key}_maps.npz') as z:
            assert np.array_equal(ids,z['ids'].astype(str))
            maps[key]={'ids':ids,'matched':z['H_matched'],'wrong':z['H_wrong'],'wrong_indices':z['wrong_indices']}
    with np.load(outdir/'object_prior.npz') as z:prior=z['maps']
    with np.load(T.ROOT/f'chuagnxindian/4.1_selective_fusion_capacity_evidence_probe/results/{ds}_144k/raw_maps.npz') as z:
        original_img=z['IMG_L4'].reshape(len(ids),49)
    rows=[];cf=[];six=[]
    for name,mm in maps.items():
        cf.append(A.counterfactual_metrics(ds,name,mm))
        for start in range(0,len(ids),32):
            end=min(start+32,len(ids));raw,score=A.resize_and_normalize(mm['matched'][start:end],device)
            _,baseline=A.resize_and_normalize(maps['original_1.3g_final']['matched'][start:end],device)
            _,obj=A.resize_and_normalize(prior[start:end],device)
            _,im=A.resize_and_normalize((original_img if name=='original_1.3g_final' else img)[start:end],device)
            for j,i in enumerate(range(start,end)):
                m=score[j];g=gt[i];gb=g>=.5;pred=m>=.6;ref=baseline[j]>=.6
                gi=fixed['indices'][i,0,:int(fixed['counts'][i,0])];ni=fixed['indices'][i,2,:int(fixed['counts'][i,2])]
                gv=m.ravel()[gi];nv=m.ravel()[ni];inter=float((pred*g).sum())
                removed=ref&~pred;added=pred&~ref
                native=mm['matched'][i].astype(np.float64);native=native/max(native.sum(),1e-30)
                entropy=float(-(native*np.log(np.maximum(native,1e-30))).sum())
                row=dict(dataset=ds,model=name,sample_id=ids[i],sample_index=i,
                    gt_response=float(gv.mean()) if len(gv) else np.nan,
                    nearfp_response=float(nv.mean()) if len(nv) else np.nan,
                    gt_nearfp_gap=float(gv.mean()-nv.mean()) if len(gv) and len(nv) else np.nan,
                    gt_vs_nearfp_auroc=A.auc_1d(gv,nv),
                    coverage=inter/max(float(g.sum()),1e-12),precision=inter/max(float(pred.sum()),1e-12),
                    background_response=float(m[~gb].mean()) if (~gb).any() else np.nan,
                    pred_area_ratio=float(pred.mean()),fp_area_ratio=float((pred&~gb).mean()),
                    RemoveFP=int((removed&~gb).sum()),RemoveTP=int((removed&gb).sum()),
                    AddTP=int((added&gb).sum()),AddFP=int((added&~gb).sum()),
                    entropy=entropy,normalized_entropy=entropy/np.log(len(native)),
                    iou=A.official_iou(pred,g),raw_range=float(raw[j].max()-raw[j].min()))
                rows.append(row)
                readouts={'AUD':m,'IMG_QUERY':im[j],'OBJ_PRIOR':obj[j],
                    'IQR':normalize(.6*m+.4*im[j]),'OGL':normalize(.6*m+.4*obj[j]),
                    'EXTRA_IQR_OGL':normalize(.6*m+.2*im[j]+.2*obj[j])}
                six.append(dict(dataset=ds,model=name,sample_id=ids[i],**{k:A.official_iou(v>=.6,g) for k,v in readouts.items()}))
        print(ds,'analyzed',name,flush=True)
    df=pd.DataFrame(rows);cf=pd.concat(cf);six=pd.DataFrame(six)
    df.to_csv(outdir/'object_per_sample.csv',index=False);cf.to_csv(outdir/'audio_swap_per_sample.csv',index=False);six.to_csv(outdir/'six_readout_per_sample.csv',index=False)
    thresholds=np.arange(21)*.05;summary=[];perf=[];dist=[];deltas=[]
    metric_cols=[k for k in df.select_dtypes(include='number').columns if k!='sample_index']
    for name in maps:
        select=df[df.model.eq(name)].sort_values('sample_index');auc=cf[cf.model.eq(name)]
        summary.append(dict(dataset=ds,model=name,**select[metric_cols].mean().to_dict(),
            cIoU=float((select.iou>=.5).mean()),AUC=float(np.trapezoid([(select.iou>=t).mean() for t in thresholds],thresholds)),
            swap_rho=float(auc.audio_swap_spearman.mean()),swap_mae=float(auc.audio_map_mae.mean())))
        for col in ('AUD','IMG_QUERY','IQR','OBJ_PRIOR','OGL','EXTRA_IQR_OGL'):
            v=six[six.model.eq(name)][col].to_numpy()
            perf.append(dict(dataset=ds,model=name,readout=col,cIoU=float((v>=.5).mean()),AUC=float(np.trapezoid([(v>=t).mean() for t in thresholds],thresholds))))
        for col in metric_cols:
            dist.append(dict(dataset=ds,model=name,metric=col,**A.stats(select[col].to_numpy(),ds,name,col)))
        for refname in ('cram_c3_stage2','short_step000'):
            ref=df[df.model.eq(refname)].sort_values('sample_index')
            assert np.array_equal(select.sample_id,ref.sample_id)
            for col in ('gt_nearfp_gap','gt_vs_nearfp_auroc','coverage','precision','pred_area_ratio','gt_response','nearfp_response'):
                deltas.append(dict(dataset=ds,model=name,reference=refname,metric=col,**A.stats(select[col].to_numpy()-ref[col].to_numpy(),ds,name,refname,col)))
    pd.DataFrame(summary).to_csv(outdir/'mechanism_summary.csv',index=False)
    pd.DataFrame(perf).to_csv(outdir/'six_readout_summary.csv',index=False)
    pd.DataFrame(dist).to_csv(outdir/'distributions.csv',index=False)
    pd.DataFrame(deltas).to_csv(outdir/'paired_deltas.csv',index=False)
    # Regress existing AUD sample metrics exactly before reporting new model results.
    old=pd.read_csv(EXP/f'mechanism_audit/results/{ds}/object_spatial_per_sample.csv',dtype={'sample_id':str})
    errors={}
    for name in A.MODELS:
        a=df[df.model.eq(name)].sort_values('sample_index');b=old[old.model.eq(name)].sort_values('sample_index')
        assert np.array_equal(a.sample_id,b.sample_id)
        errors[name]=float(np.nanmax(np.abs(a.iou.to_numpy()-b.iou.to_numpy())))
        assert errors[name]<1e-6
    save_json(outdir/'validation.json',dict(reference_iou_max_errors=errors,sample_count=len(ids),complete=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--analyze-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);torch.cuda.set_device(args.gpu);dev=torch.device('cuda',args.gpu)
    outdir=HERE/'degradation_path'/args.dataset
    if not args.analyze_only:outdir.mkdir(exist_ok=False);extract(args.dataset,dev,outdir)
    analyze(args.dataset,dev,outdir)

if __name__=='__main__':main()
