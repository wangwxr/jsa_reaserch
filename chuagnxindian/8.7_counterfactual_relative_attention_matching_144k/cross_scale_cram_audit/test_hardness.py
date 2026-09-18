#!/usr/bin/env python3
"""Evaluation-only CRAM distance audit on the frozen test split.

It supplies the exact fixed Experiment-8.1 wrong-audio assignments to the
selected *Mean-CRAM Stage-1 teacher*.  Test annotations are read only after
distance computation by the summarizer; they never enter this script.
"""
from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXP81 = ROOT / "chuagnxindian/8.1_same_image_audio_counterfactual_audit"
EXP86 = ROOT / "chuagnxindian/8.6_loss_to_decision_causal_ablation"

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod

def main():
    p = argparse.ArgumentParser(); p.add_argument('--scale', choices=('10k','144k'), required=True); p.add_argument('--dataset', choices=('vggss','flickr'), required=True); p.add_argument('--gpu', type=int, required=True); a=p.parse_args()
    evaluator = load('cross_scale_eval_loader', HERE / 'evaluate_existing.py')
    objective = load('cross_scale_objective', HERE.parent / 'objective.py')
    torch.cuda.set_device(a.gpu); device=torch.device(f'cuda:{a.gpu}')
    _C, wrapper, teacher, student, _checkpoint = evaluator.build(a.dataset, a.scale, 'mean', device)
    model = wrapper.teacher.eval()
    cache = load('cross_scale_test_cache', EXP86 / 'common.py'); data=cache.test_dataset(a.dataset)
    ids=np.asarray([Path(x).stem for x in data.image_files]).astype(str)
    with np.load(EXP81/f'results/{a.dataset}_144k_counterfactual_indices.npz') as z:
        if not np.array_equal(ids,z['anchor_ids'].astype(str)): raise RuntimeError('fixed test IDs mismatch')
        wrong=z['set1_indices'][:,:4].astype(np.int64)
    loader=DataLoader(data,batch_size=64 if a.dataset=='vggss' else 32,shuffle=False,num_workers=8,pin_memory=True,persistent_workers=True)
    rows=[]; offset=0
    # Audio is fetched from the immutable NPY-backed test dataset.  This avoids
    # any use of annotations and preserves the fixed different-sample swaps.
    for image,spec,_gt,names,_ in loader:
        b=len(names); indices=np.arange(offset,offset+b)
        wrong_spec=torch.stack([data[int(j)][1] for row in wrong[indices] for j in row]).reshape(b,4,*spec.shape[1:])
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            output=objective.forward_components(model,image.to(device,non_blocking=True).float(),spec.to(device,non_blocking=True).float())
            target=output['v2v_prob'][:,0].detach(); dpos=F.mse_loss(output['a2v_prob'][:,0],target,reduction='none').mean(1)
            ds=[]
            for k in range(4):
                wrong_attention=objective._wrong_a2v(model,wrong_spec[:,k].to(device,non_blocking=True).float(),output['visual_keys'])
                ds.append(F.mse_loss(wrong_attention,target,reduction='none').mean(1))
            dneg=torch.stack(ds,1).float().cpu().numpy(); dpos=dpos.float().cpu().numpy()
        for local,sid in enumerate(names):
            rows.append({'scale':a.scale,'dataset':a.dataset,'sample_index':int(indices[local]),'sample_id':str(sid),'d_pos':float(dpos[local]), **{f'd_neg_{k+1}':float(dneg[local,k]) for k in range(4)}})
        offset+=b; print(f'{a.scale}/{a.dataset} {offset}/{len(ids)}',flush=True)
    out=HERE/'hardness_signal'; out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows); neg=df[[f'd_neg_{k}' for k in range(1,5)]].to_numpy();
    df['D_min']=neg.min(1);df['D_mean']=neg.mean(1);df['D_max']=neg.max(1);df['D_std']=neg.std(1);df['mean_minus_min']=df.D_mean-df.D_min;df['pos_minus_min']=df.d_pos-df.D_min;df['pos_minus_mean']=df.d_pos-df.D_mean;df['negative_cv']=df.D_std/np.maximum(df.D_mean,1e-12)
    # Report both historical default soft-min and an empirical N_eff=2.5 tau.
    known={'vggss':2.6128406021282223e-05,'flickr':8.700235267679427e-06}[a.dataset]
    weights=np.exp(-(neg-neg.min(1,keepdims=True))/known);weights/=weights.sum(1,keepdims=True)
    df['weight_max']=weights.max(1);df['N_eff']=1/(weights**2).sum(1);df['weight_entropy']=-(weights*np.log(np.maximum(weights,1e-12))).sum(1)
    df.to_csv(out/f'test_distances_{a.scale}_{a.dataset}.csv',index=False)
    wrapper.close()

if __name__=='__main__': main()
