#!/usr/bin/env python3
"""Read-only M=32 counterfactual-pool audit for frozen Mean-CRAM Stage-1.

The first four candidates are the existing Experiment-8.1 set1 random-4.
Twenty-eight further candidates are sampled uniformly without replacement
from the same different-video evaluation pool, producing a nested M=32 set.
No labels, GT/OGL, regions, or residual groups participate in selection.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE=Path(__file__).resolve().parent; EXP=HERE.parent; ROOT=HERE.parents[2]
EXP81=ROOT/'chuagnxindian/8.1_same_image_audio_counterfactual_audit'; EXP86=ROOT/'chuagnxindian/8.6_loss_to_decision_causal_ablation'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def video_id(value,dataset):
    return value[:11] if dataset=='vggss' else value
def stable_seed(*items):
    return int.from_bytes(hashlib.sha256('::'.join(map(str,items)).encode()).digest()[:8],'little')%(2**32)
def make_pool(ids,dataset,random4):
    videos=np.asarray([video_id(str(x),dataset) for x in ids]);out=np.empty((len(ids),32),np.int32)
    for i in range(len(ids)):
        prefix=random4[i]; eligible=np.flatnonzero(videos!=videos[i]); eligible=eligible[~np.isin(eligible,prefix)]
        rng=np.random.default_rng(stable_seed('negative_quality_M32_tail_v1',dataset,i))
        out[i,:4]=prefix;out[i,4:]=rng.choice(eligible,size=28,replace=False)
    if np.any(videos[out]==videos[:,None]) or np.any(np.diff(np.sort(out,axis=1),axis=1)==0):raise RuntimeError('pool exclusion/uniqueness failed')
    return out

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--scale',choices=('10k','144k'),required=True);p.add_argument('--dataset',choices=('vggss','flickr'),required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    out=HERE/'candidate_pool';out.mkdir(parents=True,exist_ok=True);target=out/f'per_sample_pool_stats_{a.scale}_{a.dataset}.csv'
    if target.exists():raise RuntimeError(f'refusing to overwrite {target}')
    evaluator=load('negative_quality_eval',EXP/'cross_scale_cram_audit/evaluate_existing.py'); objective=load('negative_quality_objective',EXP/'objective.py')
    torch.cuda.set_device(a.gpu);device=torch.device(f'cuda:{a.gpu}');_C,wrapper,teacher_path,student_path,checkpoint=evaluator.build(a.dataset,a.scale,'mean',device);model=wrapper.teacher.eval()
    cache=load('negative_quality_cache',EXP86/'common.py');data=cache.test_dataset(a.dataset);ids=np.asarray([Path(x).stem for x in data.image_files]).astype(str);n=len(ids)
    with np.load(EXP81/f'results/{a.dataset}_144k_counterfactual_indices.npz') as z:
        if not np.array_equal(ids,z['anchor_ids'].astype(str)):raise RuntimeError('8.1 ID mismatch')
        random4=z['set1_indices'][:,:4].astype(np.int32)
    pool=make_pool(ids,a.dataset,random4)
    bs=64 if a.dataset=='vggss' else 32;loader=DataLoader(data,batch_size=bs,shuffle=False,num_workers=8,pin_memory=True,persistent_workers=True)
    # In eval mode _masked is identity. Cache the exact existing audio-branch
    # query once, then reuse it for all M candidate pairings.
    queries=np.empty((n,2,512),np.float32); labels=np.empty(n,object);pos=0
    for _frame,spec,_gt,names,label in loader:
        audio=spec.to(device,non_blocking=True).float();tokens=model._audio_tokens(model.audnet(audio));slots=model.slot_attn.slots.expand(len(audio),-1,-1);_s,q,_k=model.slot_attn.audio_branch(tokens,slots)
        queries[pos:pos+len(names)]=q.cpu().numpy();labels[pos:pos+len(names)]=list(label);pos+=len(names)
    rows=[];candidate_rows=[];pos=0;query_replay=0.
    for frame,spec,_gt,names,_label in loader:
        b=len(names);ix=np.arange(pos,pos+b);frame=frame.to(device,non_blocking=True).float();spec=spec.to(device,non_blocking=True).float()
        output=objective.forward_components(model,frame,spec);target_v=output['v2v_prob'][:,0].detach();dpos=F.mse_loss(output['a2v_prob'][:,0],target_v,reduction='none').mean(1)
        query_replay=max(query_replay,float((output['audio_query']-torch.from_numpy(queries[ix]).to(device)).abs().max()))
        candidate_q=torch.from_numpy(queries[pool[ix]]).to(device);keys=output['visual_keys'][:,None].expand(-1,32,-1,-1).reshape(b*32,49,512);qflat=candidate_q.reshape(b*32,2,512)
        _logits,prob=objective.attention(qflat,keys);prob=prob[:,0].reshape(b,32,49);dist=F.mse_loss(prob,target_v[:,None],reduction='none').mean(2).float().cpu().numpy();dpos_np=dpos.float().cpu().numpy()
        for j,sid in enumerate(names):
            d=dist[j];top=np.argsort(d)[:4];r4=d[:4]
            row={'scale':a.scale,'dataset':a.dataset,'sample_index':int(ix[j]),'sample_id':str(sid),'label':str(labels[ix[j]]),'d_pos':float(dpos_np[j]),'current_random4_mean':float(r4.mean()),'current_random4_min':float(r4.min()),'pool_min':float(d.min()),'pool_mean':float(d.mean()),'pool_std':float(d.std()),'pool_P10':float(np.quantile(d,.1)),'pool_P25':float(np.quantile(d,.25)),'top4_mean':float(d[top].mean()),'top4_min':float(d[top].min()),'top4_std':float(d[top].std()),'opportunity_gap_mean':float(r4.mean()-d[top].mean()),'opportunity_gap_min':float(r4.min()-d.min()),'violation_fraction_pool':float((d<dpos_np[j]).mean()),'random4_violation_fraction':float((r4<dpos_np[j]).mean()),'missed_confusing_negative':bool(d.min()<dpos_np[j] and r4.min()>=dpos_np[j]),'top4_vs_random4_ratio':float(d[top].mean()/max(r4.mean(),1e-12)),'top4_pool_indices':json.dumps(pool[ix[j],top].tolist()),'top4_pool_ids':json.dumps(ids[pool[ix[j],top]].tolist()),'top4_pool_labels':json.dumps(labels[pool[ix[j],top]].astype(str).tolist())}
            rows.append(row)
            for rank,k in enumerate(top):candidate_rows.append({'scale':a.scale,'dataset':a.dataset,'sample_id':str(sid),'sample_index':int(ix[j]),'rank':rank+1,'candidate_index':int(pool[ix[j],k]),'candidate_id':str(ids[pool[ix[j],k]]),'candidate_label':str(labels[pool[ix[j],k]]),'distance':float(d[k]),'is_same_label':bool(labels[ix[j]]==labels[pool[ix[j],k]]),'is_random4':bool(k<4)})
        pos+=b;print(f'{a.scale}/{a.dataset}: {pos}/{n}',flush=True)
    pd.DataFrame(rows).to_csv(target,index=False);pd.DataFrame(candidate_rows).to_csv(out/f'top4_candidates_{a.scale}_{a.dataset}.csv',index=False)
    np.savez_compressed(out/f'pool_indices_{a.scale}_{a.dataset}.npz',ids=ids,random4_indices=random4,pool_indices=pool)
    (out/f'audit_{a.scale}_{a.dataset}.json').write_text(json.dumps({'read_only':True,'scale':a.scale,'dataset':a.dataset,'M':32,'K_random':4,'random4_source':'Experiment 8.1 set1 prefix','tail_seed_scheme':'sha256(negative_quality_M32_tail_v1,dataset,sample_index)','different_video':True,'query_replay_max_abs_error':query_replay,'teacher':str(teacher_path),'teacher_epoch':int(checkpoint['epoch'])},indent=2)+'\n')
    wrapper.close()
if __name__=='__main__':main()
