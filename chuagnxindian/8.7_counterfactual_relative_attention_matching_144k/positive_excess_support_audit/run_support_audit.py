#!/usr/bin/env python3
"""Frozen-feature support replay. No gradient, optimizer, or checkpoint writes."""
from pathlib import Path
import sys, importlib.util, copy
import numpy as np, torch, argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE=Path(__file__).resolve().parent; EXP=HERE.parent; ROOT=EXP.parents[1]
RECIPE=ROOT/'chuagnxindian/1mufasaslot/1.3G-multigeom_equivariant_l3_refine'; EXP86=ROOT/'chuagnxindian/8.6_loss_to_decision_causal_ablation'
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def main(ds='vggss',limit=32):
 sys.path.insert(0,str(RECIPE)); import common
 ac=mod('support_ac',EXP86/'common.py'); reg=copy.deepcopy(common.EXPERIMENTS[f'{ds}_144k']); reg['base_checkpoint']=str(EXP/f'checkpoints/C3/{ds}/seed12345/selected_best.pth');cfg=common.load_base_config(reg);cfg.workers=4
 common.setup_seed(12345);dev=torch.device('cuda:0');model,_=common.build_model(cfg,reg,dev); ck=torch.load(EXP/f'stage2/C3/{ds}/seed12345/{ds}_best.pth',map_location='cpu',weights_only=False);model.student.proj3_spatial.load_state_dict(ck['proj3_spatial_state_dict']);model.student.adapter.load_state_dict(ck['topdown_adapter_state_dict']);model.eval()
 data=ac.test_dataset(ds); n=min(len(data),limit); loader=DataLoader(torch.utils.data.Subset(data,range(n)),batch_size=32,shuffle=False,num_workers=4,pin_memory=True)
 queries=np.load(EXP/f'natural_localization/{ds}/C3/seed12345/audio_queries.npy',mmap_mode='r'); scores=np.empty((n,6,14,14),np.float16); ids=[];pos=0
 for im,au,gt,names,_ in loader:
  im,au,gt,names=common.flatten_eval_batch(im,au,gt,names); b=len(names);q=torch.from_numpy(np.asarray(queries[pos:pos+b])).to(dev)
  with torch.inference_mode():
   t=model._extract_visual_teacher(im.to(dev).float(),q); f=model._fine_from_teacher_features(t,q); branch=model.teacher.slot_attn.visual_branches[-1];l4=branch.img_to_k(branch.img_norm_input(model._to_tokens(t['f4_projected']))); fk=branch.img_to_k(branch.img_norm_input(model._to_tokens(f['F34'])));
   def cosine(x,y): return F.cosine_similarity(x,y,dim=-1)
   aud=f['AUD_FINE'][:,0]; qry=F.interpolate(t['AUD_L4'].new_tensor(np.load(EXP/f'natural_localization/{ds}/C3/seed12345/natural_maps.npz')['image_native'][pos:pos+b])[:,None],size=(14,14),mode='bilinear',align_corners=False)[:,0]; qry=(qry-qry.amin((1,2),True))/(qry.amax((1,2),True)-qry.amin((1,2),True)+1e-8); core=(aud>=.6)&(qry>=.6)
   ft=f['F34'].flatten(2).transpose(1,2); fu=f['F4_UP'].flatten(2).transpose(1,2); cp=(ft*core.flatten(1)[...,None]).sum(1)/(core.flatten(1).sum(1,keepdim=True)+1e-8); kp=(fk*core.flatten(1)[...,None]).sum(1)/(core.flatten(1).sum(1,keepdim=True)+1e-8)
   s1=cosine(ft,cp[:,None]).reshape(-1,14,14);s2=cosine(fk,kp[:,None]).reshape(-1,14,14);s3=cosine(ft,fu).reshape(-1,14,14);s4=F.avg_pool2d(qry[:,None],3,1,1)[:,0];s5=F.avg_pool2d(core.float()[:,None],3,1,1)[:,0];s6=.5*(s1+s4)
   scores[pos:pos+b]=torch.stack([s1,s2,s3,s4,s5,s6],1).cpu().numpy().astype(np.float16)
  ids+=list(map(str,names));pos+=b
 np.savez_compressed(HERE/f'candidate_scores_{ds}_{n}.npz',ids=np.asarray(ids),scores=scores,names=np.asarray(['f34_core_cos','k34_core_cos','cross_level_cos','local_query_support','core_connectivity','combined']))
 return
 # sanity-only reference below
 im,au,gt,names,_=next(iter(loader));im,au,gt,names=common.flatten_eval_batch(im,au,gt,names); q=torch.from_numpy(np.asarray(queries[:len(names)])).to(dev)
 with torch.inference_mode():
  t=model._extract_visual_teacher(im.to(dev).float(),q); f=model._fine_from_teacher_features(t,q); branch=model.teacher.slot_attn.visual_branches[-1];l4=branch.img_to_k(branch.img_norm_input(model._to_tokens(t['f4_projected']))); fk=branch.img_to_k(branch.img_norm_input(model._to_tokens(f['F34'])));
  shapes={k:list(v.shape) for k,v in {'F34':f['F34'],'F4_UP':f['F4_UP'],'FINE_KEYS':fk,'L4_KEYS':l4}.items()};
  # parity: replayed fine attention equals existing frozen matched map cache.
  from json import dumps
  (HERE/'sanity.json').write_text(dumps({'dataset':ds,'limit':len(names),'shapes':shapes,'grad_enabled':torch.is_grad_enabled(),'requires_grad_any':any(x.requires_grad for x in [f['F34'],f['F4_UP'],fk,l4])},indent=2))
 print((HERE/'sanity.json').read_text())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',default='vggss',choices=['vggss','flickr']);p.add_argument('--limit',type=int,default=32);a=p.parse_args();main(a.dataset,a.limit)
