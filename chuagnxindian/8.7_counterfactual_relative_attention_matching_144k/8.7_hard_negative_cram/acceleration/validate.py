#!/usr/bin/env python3
"""Post-Stage-2 exactness and speed validation of eval-audio-token reuse.

This is a no-checkpoint, no-dataset-write experiment.  It compares the active
Hard-CRAM implementation with a candidate that evaluates each batch audio once
in audnet.eval() and gathers its tokens for K=4 wrong-audio branches.
"""
from __future__ import annotations
import argparse, copy, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

HERE=Path(__file__).resolve().parent;HARD=HERE.parent;MEAN=HARD.parent
sys.path.insert(0,str(HARD));import common, objective

def eval_audio_tokens(model,audio):
    was=model.audnet.training;model.audnet.eval()
    try:return model._audio_tokens(model.audnet(audio))
    finally:model.audnet.train(was)
def wrong_from_tokens(model,tokens,keys):
    slots=model.slot_attn.slots.expand(len(tokens),-1,-1)
    masked=model.slot_attn._masked(tokens,model.slot_attn.mask_token_aud)
    _slots,query,_keys=model.slot_attn.audio_branch(masked,slots)
    _logits,prob=objective.attention(query,keys)
    return prob[:,0]
def reuse_components(model,out,spec,offsets,tau):
    keys=out['visual_keys'];target=out['v2v_prob'][:,0].detach()
    # One eval-mode encoder for every source audio. Reentrant checkpoint retains
    # the original memory policy and recomputes this encoder only once backward.
    tokens=checkpoint(lambda item:eval_audio_tokens(model,item),spec.detach().requires_grad_(True),use_reentrant=True)
    pieces=[]
    for k in range(offsets.shape[1]):
        selected=tokens[offsets[:,k]]
        attn=checkpoint(lambda item,visual_keys:wrong_from_tokens(model,item,visual_keys),selected,keys,use_reentrant=True)
        pieces.append(F.mse_loss(attn,target,reduction='none').mean(1))
    individual=torch.stack(pieces,1);dneg,weights=objective.aggregate_negative(individual,'softmin',tau);dpos=F.mse_loss(out['a2v_prob'][:,0],target,reduction='none').mean(1)
    return dict(d_pos=dpos.mean(),d_neg=dneg.mean(),d_neg_individual=individual,cram_loss=F.softplus(dpos-dneg).mean(),weights=weights)
def build(dataset,gpu,run_root):
    common.setup_seed(12345);device=torch.device(f'cuda:{gpu}');init=common.initialization_path(dataset,12345);model=common.build_model(dataset,device,init,trainable=True)
    ck=run_root/f'stage1/{dataset}/seed12345/selected_best.pth';payload=torch.load(ck,map_location='cpu',weights_only=False);model.load_state_dict(payload['model'],strict=True)
    config=json.loads((run_root/f'stage1/{dataset}/seed12345/hard_cram_config.json').read_text());return model,payload,config,device
def batch(dataset,device,batch_size):
    orders=np.load(common.order_path(dataset,12345),mmap_mode='r');neg=np.load(MEAN/f'configs/{dataset}_seed12345_negative_offsets.npy',mmap_mode='r');data=common.train_dataset(dataset);sampler=common.EpochOrderSampler(orders);sampler.set_epoch(0)
    frame,spec,*_=next(iter(DataLoader(data,batch_size=batch_size,sampler=sampler,num_workers=0,drop_last=True)))
    if batch_size == common.BATCH_SIZE:
        offsets=np.array(neg[0,:batch_size],dtype=np.int64,copy=True)
        mapping='frozen_8.7_batch_offsets'
    else:
        # A self-contained K=4 derangement for the low-memory profiler only.
        # Formal 256-batch validation always uses the frozen training mapping.
        offsets=np.stack([(np.arange(batch_size)+k+1)%batch_size for k in range(4)],axis=1)
        mapping='cyclic_derangement_microbenchmark'
    return frame.to(device).float(),spec.to(device).float(),torch.as_tensor(offsets,device=device),mapping
def one(model,state,frame,anchor_spec,source_spec,offsets,tau,mode,seed,grad):
    model.load_state_dict(state,strict=True);model.train();model.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda'):
        with torch.random.fork_rng(devices=[frame.device.index]):
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);out=objective.forward_components(model,frame,anchor_spec)
            cram=objective.cram_components(model,out,source_spec[offsets], 'softmin',tau) if mode=='reference' else reuse_components(model,out,source_spec,offsets,tau)
    if grad:cram['cram_loss'].backward()
    return cram,{n:p.grad.detach().float().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('vggss','flickr'),required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--run-root',type=Path,default=HARD);p.add_argument('--steps',type=int,default=12);p.add_argument('--batch-size',type=int,default=common.BATCH_SIZE);p.add_argument('--output-name',default='acceleration_validation.json');a=p.parse_args();run_root=a.run_root.resolve()
    if not 4 <= a.batch_size <= common.BATCH_SIZE: raise ValueError('--batch-size must be in [4,256]')
    out=run_root/f'acceleration/{a.dataset}';out.mkdir(parents=True,exist_ok=True)
    model,payload,cfg,device=build(a.dataset,a.gpu,run_root);tau=float(cfg['cram_softmin_tau']);frame,spec,offsets,mapping=batch(a.dataset,device,a.batch_size);state=copy.deepcopy(model.state_dict())
    # Exactness uses a 64-anchor subset while retaining the full source-audio
    # batch for valid frozen offsets; it avoids artificially changing negatives.
    n=min(64,a.batch_size);small_frame,small_anchor_spec,small_offsets=frame[:n],spec[:n],offsets[:n]
    ref,gr=one(model,state,small_frame,small_anchor_spec,spec,small_offsets,tau,'reference',911001,True);opt,go=one(model,state,small_frame,small_anchor_spec,spec,small_offsets,tau,'reuse',911001,True)
    # Final D_neg, loss, and all trainable-parameter gradients jointly verify
    # every negative branch without changing the formal training return API.
    exact={'d_pos_max_abs':float((ref['d_pos']-opt['d_pos']).abs()),'d_neg_max_abs':float((ref['d_neg']-opt['d_neg']).abs()),'loss_max_abs':float((ref['cram_loss']-opt['cram_loss']).abs())}
    dot=nr=no=0.;max_abs=0.
    for name in gr:
        x,y=gr[name],go[name];dot+=float((x*y).sum());nr+=float(x.square().sum());no+=float(y.square().sum());max_abs=max(max_abs,float((x-y).abs().max()))
    exact|={'gradient_cosine':dot/max((nr*no)**.5,1e-30),'gradient_relative_l2':((nr+no-2*dot)**.5)/max(nr**.5,1e-30),'gradient_max_abs':max_abs}
    # Compute-only training-step throughput on the true 256-anchor batch.
    speed={}
    for mode in ('reference','reuse'):
        model.load_state_dict(state,strict=True);model.train();optim=torch.optim.AdamW(model.parameters(),lr=common.LR,weight_decay=common.WEIGHT_DECAY);scaler=torch.amp.GradScaler('cuda');times=[]
        for step in range(a.steps+2):
            optim.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
            with torch.amp.autocast('cuda'):
                with torch.random.fork_rng(devices=[a.gpu]):
                    torch.manual_seed(912000+step);torch.cuda.manual_seed_all(912000+step);output=objective.forward_components(model,frame,spec)
                    cram=objective.cram_components(model,output,spec[offsets],'softmin',tau) if mode=='reference' else reuse_components(model,output,spec,offsets,tau)
                    loss=objective.total_loss(output,cram['cram_loss'],1.0)
            scaler.scale(loss).backward();scaler.step(optim);scaler.update();torch.cuda.synchronize()
            if step>=2:times.append(time.perf_counter()-start)
        speed[mode]={'mean_step_seconds':float(np.mean(times)),'median_step_seconds':float(np.median(times)),'steps':a.steps}
    speed['speedup_reference_over_reuse']=speed['reference']['mean_step_seconds']/speed['reuse']['mean_step_seconds']
    passed=all(v<2e-6 for k,v in exact.items() if 'abs' in k) and exact['gradient_cosine']>.9999 and exact['gradient_relative_l2']<2e-4 and speed['speedup_reference_over_reuse']>1.0
    result={'dataset':a.dataset,'teacher':str(run_root/f'stage1/{a.dataset}/seed12345/selected_best.pth'),'teacher_epoch':int(payload['epoch']),'tau':tau,'K':4,'batch_size':a.batch_size,'negative_mapping':mapping,'exactness':exact,'throughput':speed,'passed':passed,'criterion':'all attention-distance/loss abs<2e-6, grad cosine>.9999, relative L2<2e-4, speedup>1'}
    (out/a.output_name).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));model.cpu();torch.cuda.empty_cache()
    if not passed:raise SystemExit(2)
if __name__=='__main__':main()
