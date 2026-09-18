#!/usr/bin/env python3
"""Run the unchanged 1.3G vanilla Stage-2 from a completed Hard-CRAM teacher."""
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
HARD=HERE.parent
MEAN=HARD.parent
ROOT=MEAN.parents[1]
RECIPE=ROOT/'chuagnxindian/1mufasaslot/1.3G-multigeom_equivariant_l3_refine'
EXP86=ROOT/'chuagnxindian/8.6_loss_to_decision_causal_ablation'

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('vggss','flickr'),required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--run-root',type=Path,default=HARD);p.add_argument('--sanity-only',action='store_true');a=p.parse_args();run_root=a.run_root.resolve()
 teacher=run_root/f'stage1/{a.dataset}/seed12345/selected_best.pth';final=run_root/f'stage1/{a.dataset}/seed12345/final.pth';out=run_root/f'stage2/{a.dataset}/seed12345'
 if not final.is_file():raise RuntimeError(f'Stage-1 is not complete; waiting for {final}')
 if not teacher.is_file():raise FileNotFoundError(teacher)
 # A previous auto-launch can have created only its manifest before failing.
 # Preserve that evidence, but permit a clean retry.  Any actual Stage-2
 # artifact remains strictly protected.
 protected=('latest.pth','final.pth',f'{a.dataset}_best.pth','epoch_metrics.csv','configs.json')
 if out.exists() and any((out/name).exists() for name in protected):raise RuntimeError(f'Refusing to overwrite completed/active result {out}')
 if out.exists() and (out/'launch_manifest.json').is_file() and not a.sanity_only:
  os.replace(out/'launch_manifest.json',out/f'failed_launch_manifest_{int(time.time())}.json')
 checkpoint=torch.load(teacher,map_location='cpu',weights_only=False);metrics=checkpoint.get('metrics',{})
 if not {'AUD_cIoU','AUD_AUC'}<=set(metrics):raise RuntimeError('Teacher checkpoint lacks Stage-1 AUD validation metrics')
 sys.path.insert(0,str(RECIPE));import common
 trainer=load('_hard_vanilla_stage2_train',RECIPE/'train.py');cache=load('_hard_stage2_cached_eval',EXP86/'common.py')
 registry=copy.deepcopy(common.EXPERIMENTS[f'{a.dataset}_144k']);registry['expected_aud']=(float(metrics['AUD_cIoU']),float(metrics['AUD_AUC']))
 config=common.load_base_config(registry)
 if not(config.seed==12345 and config.batch_size==256 and config.epochs==50):raise RuntimeError('Unexpected vanilla Stage-2 recipe')
 def datasets(cfg,selected):
  return common.get_train_dataset(cfg,hard_img=cfg.hard_img,hard_aud=cfg.hard_aud,rand_aud=cfg.rand_aud),cache.test_dataset(selected['dataset'])
 # `train.py` obtains the registry again from common.EXPERIMENTS.  Updating
 # that exact object and base_checkpoint_path is therefore essential: merely
 # passing a copied registry would silently retain the 1.3G teacher.
 common.EXPERIMENTS[f'{a.dataset}_144k']=registry
 common.base_checkpoint_path=lambda _registry: teacher.resolve()
 common.build_datasets=datasets;trainer.build_datasets=datasets
 if not a.sanity_only:out.mkdir(parents=True,exist_ok=True)
 manifest={'method':'Hard-CRAM Stage-1 -> unchanged vanilla 1.3G Stage-2','dataset':a.dataset,'seed':12345,'gpu':a.gpu,'stage1_selected_teacher':str(teacher),'stage1_selected_sha256':sha(teacher),'stage1_final_sha256':sha(final),'teacher_epoch':int(checkpoint['epoch']),'teacher_expected_aud':registry['expected_aud'],'stage2_epochs':50,'batch_size':256,'loss':'unchanged coarse anchoring + equivariance; no CRAM','recipe':str(RECIPE),'recipe_hashes':{x.name:sha(x) for x in RECIPE.glob('*.py')},'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
 if not a.sanity_only:(out/'launch_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
 sys.argv=[str(RECIPE/'train.py'),'--experiment',f'{a.dataset}_144k','--gpu',str(a.gpu),'--epochs','50','--model-dir',str(out.parent),'--experiment-name',out.name]+(['--sanity-only'] if a.sanity_only else [])
 trainer.main()
 if sha(teacher)!=manifest['stage1_selected_sha256']:raise RuntimeError('Teacher changed during Stage-2')
if __name__=='__main__':main()
