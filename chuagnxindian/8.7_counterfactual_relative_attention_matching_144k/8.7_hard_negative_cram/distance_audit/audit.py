#!/usr/bin/env python3
"""Read-only fixed-training-batch distance audit for Mean-CRAM C3."""
from __future__ import annotations

import argparse, csv, hashlib, importlib.util, json, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE=Path(__file__).resolve().parent
EXP=HERE.parents[1]
SOURCE=EXP

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module; spec.loader.exec_module(module); return module
common=load('hard_cram_common',SOURCE/'common.py')
objective=load('hard_cram_objective',SOURCE/'objective.py')

def stats(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    return {k:float(v) for k,v in {'mean':x.mean(),'median':np.median(x),'p25':np.quantile(x,.25),'p75':np.quantile(x,.75),'p90':np.quantile(x,.90),'p95':np.quantile(x,.95)}.items()}|{'n':int(len(x))}

def effective(dist,tau):
    logits=-(dist-dist.min(1,keepdims=True))/tau
    weight=np.exp(logits);weight/=weight.sum(1,keepdims=True)
    return weight,1/(weight**2).sum(1)

def tau_for(dist,target):
    spread=np.maximum(dist.max(1)-dist.min(1),1e-14); lo=np.log(spread.min()/1e4);hi=np.log(spread.max()*1e4)
    for _ in range(80):
        mid=(lo+hi)/2; value=effective(dist,np.exp(mid))[1].mean()
        if value<target:lo=mid
        else:hi=mid
    return float(np.exp((lo+hi)/2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('vggss','flickr'),required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--batches',type=int,default=64);args=p.parse_args()
    out=HERE/args.dataset;out.mkdir(parents=True,exist_ok=True)
    torch.cuda.set_device(args.gpu);device=torch.device(f'cuda:{args.gpu}'); common.setup_seed(12345)
    init=common.initialization_path(args.dataset,12345); ckpt=SOURCE/f'checkpoints/C3/{args.dataset}/seed12345/selected_best.pth'
    model=common.build_model(args.dataset,device,init,trainable=True); payload=torch.load(ckpt,map_location='cpu',weights_only=False);model.load_state_dict(payload['model'],strict=True);model.train()
    orders=np.load(common.order_path(args.dataset,12345),mmap_mode='r');negative=np.load(SOURCE/f'configs/{args.dataset}_seed12345_negative_offsets.npy',mmap_mode='r');dataset=common.train_dataset(args.dataset)
    sampler=common.EpochOrderSampler(orders);sampler.set_epoch(0)
    loader=DataLoader(dataset,batch_size=common.BATCH_SIZE,sampler=sampler,num_workers=common.DATASETS[args.dataset]['workers'],pin_memory=True,drop_last=True,persistent_workers=False,prefetch_factor=2,generator=torch.Generator().manual_seed(12345))
    rows=[]
    for b,(frame,spec,_boxes,ids,_labels) in enumerate(loader):
        if b>=args.batches:break
        start=b*common.BATCH_SIZE; expected=[Path(dataset.image_files[int(i)]).stem for i in orders[0,start:start+len(ids)]]
        if list(map(str,ids))!=expected:raise RuntimeError(f'order mismatch batch {b}')
        frame=frame.to(device,non_blocking=True).float();spec=spec.to(device,non_blocking=True).float()
        offsets=torch.as_tensor(np.array(negative[0,start:start+len(ids)],dtype=np.int64,copy=True),device=device)
        wrong=spec[offsets]
        # Deterministic diagnostic mask state, distinct from any historical training RNG state.
        with torch.no_grad(), torch.amp.autocast('cuda'):
            with torch.random.fork_rng(devices=[args.gpu]):
                torch.manual_seed(912345+b);torch.cuda.manual_seed_all(912345+b)
                output=objective.forward_components(model,frame,spec)
                target=output['v2v_prob'][:,0].detach();dpos=F.mse_loss(output['a2v_prob'][:,0],target,reduction='none').mean(1)
                ds=[]
                for k in range(wrong.shape[1]):
                    a=objective._wrong_a2v(model,wrong[:,k],output['visual_keys'])
                    ds.append(F.mse_loss(a,target,reduction='none').mean(1))
                dneg=torch.stack(ds,1)
        data=torch.stack((dpos,dneg[:,0],dneg[:,1],dneg[:,2],dneg[:,3]),1).float().cpu().numpy()
        source_ids=np.asarray(expected)[negative[0,start:start+len(ids)]]
        for j,sid in enumerate(ids): rows.append(dict(dataset=args.dataset,epoch=0,batch=b,sample_id=str(sid),wrong_audio_1=str(source_ids[j,0]),wrong_audio_2=str(source_ids[j,1]),wrong_audio_3=str(source_ids[j,2]),wrong_audio_4=str(source_ids[j,3]),d_pos=float(data[j,0]),d_neg_1=float(data[j,1]),d_neg_2=float(data[j,2]),d_neg_3=float(data[j,3]),d_neg_4=float(data[j,4])))
        print(args.dataset,b+1,args.batches,flush=True)
    import pandas as pd
    df=pd.DataFrame(rows); neg=df[[f'd_neg_{i}' for i in range(1,5)]].to_numpy();df['d_neg_min']=neg.min(1);df['d_neg_mean']=neg.mean(1);df['d_neg_max']=neg.max(1);df['d_neg_std']=neg.std(1);df['mean_minus_min']=df.d_neg_mean-df.d_neg_min;df['min_minus_pos']=df.d_neg_min-df.d_pos;df['mean_minus_pos']=df.d_neg_mean-df.d_pos
    tau_default=tau_for(neg,2.5);tau_stronger=tau_for(neg,2.0)
    tau_rows=[]
    for name,tau in [('default_neff_2.5',tau_default),('stronger_neff_2.0',tau_stronger)]:
        w,n=effective(neg,tau); order=np.sort(w,axis=1)[:,::-1]
        tau_rows.append(dict(name=name,tau=tau,**{f'neff_{k}':v for k,v in stats(n).items()},**{f'hardest_weight_{k}':v for k,v in stats(order[:,0]).items()},**{f'second_weight_{k}':v for k,v in stats(order[:,1]).items()}))
    df.to_csv(out/'per_sample_distances.csv',index=False)
    summary={col:stats(df[col]) for col in ['d_pos','d_neg_min','d_neg_mean','d_neg_max','d_neg_std','mean_minus_min','min_minus_pos','mean_minus_pos']}
    flags=dict(min_closer_than_pos_fraction=float((df.min_minus_pos<0).mean()),mean_farther_than_pos_fraction=float((df.mean_minus_pos>0).mean()),hard_but_mean_easy_fraction=float(((df.min_minus_pos<0)&(df.mean_minus_pos>0)).mean()))
    report=dict(dataset=args.dataset,checkpoint=str(ckpt),checkpoint_epoch=int(payload['epoch']),seed=12345,epoch_order=0,batches=args.batches,batch_size=common.BATCH_SIZE,K=4,attention_shape='[B, 2, 49], slot 0 [B,49]',summary=summary,flags=flags,tau_candidates=tau_rows)
    (out/'distance_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.DataFrame(tau_rows).to_csv(out/'tau_candidates.csv',index=False)
    print(json.dumps(report,indent=2))
    model.cpu();torch.cuda.empty_cache()
if __name__=='__main__':main()
