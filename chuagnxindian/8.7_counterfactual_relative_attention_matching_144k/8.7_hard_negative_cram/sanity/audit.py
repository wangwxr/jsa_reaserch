#!/usr/bin/env python3
"""Fixed-batch Mean-vs-Hard CRAM loss and gradient-scale sanity check."""
from __future__ import annotations
import argparse,copy,csv,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
HERE=Path(__file__).resolve().parent;EXP=HERE.parent;sys.path.insert(0,str(EXP))
import common,objective
SOURCE=EXP.parent
def grad_vector(model):
 x=[p.grad.detach().float().flatten() for p in model.parameters() if p.grad is not None]
 return torch.cat(x)
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('vggss','flickr'),required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--tau',type=float,required=True);p.add_argument('--batches',type=int,default=4);p.add_argument('--microbatch',type=int,default=64);a=p.parse_args();out=HERE/a.dataset;out.mkdir(parents=True,exist_ok=True)
 torch.cuda.set_device(a.gpu);d=torch.device(f'cuda:{a.gpu}');common.setup_seed(12345);init=common.initialization_path(a.dataset,12345);model=common.build_model(a.dataset,d,init,trainable=True);ck=torch.load(SOURCE/f'checkpoints/C3/{a.dataset}/seed12345/selected_best.pth',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);initial=copy.deepcopy(model.state_dict());orders=np.load(common.order_path(a.dataset,12345),mmap_mode='r');neg=np.load(SOURCE/f'configs/{a.dataset}_seed12345_negative_offsets.npy',mmap_mode='r');data=common.train_dataset(a.dataset);sampler=common.EpochOrderSampler(orders);sampler.set_epoch(0);loader=DataLoader(data,batch_size=common.BATCH_SIZE,sampler=sampler,num_workers=common.DATASETS[a.dataset]['workers'],pin_memory=True,drop_last=True,persistent_workers=False,prefetch_factor=2)
 rows=[]
 for b,(frame,spec,*_) in enumerate(loader):
  if b>=a.batches:break
  full_spec=spec.to(d).float();frame=frame[:a.microbatch].to(d).float();spec=full_spec[:a.microbatch]
  wrong=full_spec[torch.as_tensor(np.array(neg[0,b*common.BATCH_SIZE:b*common.BATCH_SIZE+a.microbatch],dtype=np.int64,copy=True),device=d)]
  results={}
  for name,tau in [('mean',None),('softmin',a.tau)]:
   model.load_state_dict(initial);model.train();model.zero_grad(set_to_none=True)
   # FP32 is intentional: it measures the mathematical unscaled gradients.
   # The production recipe retains its original AMP GradScaler unchanged.
   with torch.amp.autocast('cuda',enabled=False):
    with torch.random.fork_rng(devices=[a.gpu]):
     torch.manual_seed(812345+b);torch.cuda.manual_seed_all(812345+b);outp=objective.forward_components(model,frame,spec);cram=objective.cram_components(model,outp,wrong,name,tau);base=objective.total_loss(outp,cram['cram_loss']*0,1.0)
   cram['cram_loss'].backward();g=grad_vector(model);results[name]=(float(base.detach()),float(cram['cram_loss'].detach()),g,float(g.norm()),float(g.abs().max()),float(g.abs().mean()),float(cram['effective_negative_number'].mean()),float(cram['hardest_weight'].mean()))
  cosine=float(torch.dot(results['mean'][2],results['softmin'][2])/max(results['mean'][3]*results['softmin'][3],1e-30));rows.append(dict(batch=b,base_loss_mean=results['mean'][0],base_loss_hard=results['softmin'][0],mean_cram_loss=results['mean'][1],hard_cram_loss=results['softmin'][1],mean_grad_norm=results['mean'][3],hard_grad_norm=results['softmin'][3],hard_to_mean_grad_ratio=results['softmin'][3]/max(results['mean'][3],1e-30),gradient_cosine=cosine,mean_max_parameter_grad=results['mean'][4],hard_max_parameter_grad=results['softmin'][4],mean_mean_parameter_grad=results['mean'][5],hard_mean_parameter_grad=results['softmin'][5],hard_neff=results['softmin'][6],hardest_weight=results['softmin'][7]))
 import pandas as pd
 df=pd.DataFrame(rows);df.to_csv(out/'sanity_per_batch.csv',index=False);summary={c:float(df[c].mean()) for c in df.columns if c!='batch'}|{'dataset':a.dataset,'tau':a.tau,'batches':a.batches,'microbatch':a.microbatch,'finite':bool(np.isfinite(df.select_dtypes('number').to_numpy()).all())};(out/'sanity_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
