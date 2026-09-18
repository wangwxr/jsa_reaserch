#!/usr/bin/env python3
"""Cache frozen Mean-CRAM Stage-1 audio queries for audit-only diversity metrics."""
from __future__ import annotations
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
HERE=Path(__file__).resolve().parent;EXP=HERE.parent;ROOT=HERE.parents[2]
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--scale',choices=('10k','144k'),required=True);p.add_argument('--dataset',choices=('vggss','flickr'),required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();out=HERE/'diversity'/f'audio_queries_{a.scale}_{a.dataset}.npz'
 if out.exists():raise RuntimeError(f'refusing to overwrite {out}')
 e=load('nq_eval',EXP/'cross_scale_cram_audit/evaluate_existing.py');c=load('nq_cache',ROOT/'chuagnxindian/8.6_loss_to_decision_causal_ablation/common.py');torch.cuda.set_device(a.gpu);dev=torch.device(f'cuda:{a.gpu}');_C,w,_t,_s,_k=evaluator=e.build(a.dataset,a.scale,'mean',dev);m=w.teacher.eval();d=c.test_dataset(a.dataset);loader=DataLoader(d,batch_size=128 if a.dataset=='vggss' else 32,shuffle=False,num_workers=8,pin_memory=True,persistent_workers=True);qs=[];ids=[];labels=[]
 for _f,spec,_g,names,label in loader:
  audio=spec.to(dev,non_blocking=True).float();tok=m._audio_tokens(m.audnet(audio));slots=m.slot_attn.slots.expand(len(audio),-1,-1);_x,q,_y=m.slot_attn.audio_branch(tok,slots);qs.append(q.cpu().numpy());ids.extend(map(str,names));labels.extend(map(str,label));print(a.scale,a.dataset,len(ids),flush=True)
 np.savez_compressed(out,ids=np.asarray(ids),labels=np.asarray(labels),queries=np.concatenate(qs));w.close()
if __name__=='__main__':main()
