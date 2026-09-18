import sys,copy,importlib.util,json
from pathlib import Path
import torch,numpy as np
from torch.utils.data import DataLoader,Subset
R=Path(__file__).resolve().parents[1];E=R.parent;ROOT=E.parents[1];RECIPE=ROOT/'chuagnxindian/1mufasaslot/1.3G-multigeom_equivariant_l3_refine';EXP86=ROOT/'chuagnxindian/8.6_loss_to_decision_causal_ablation'
def load(n,p):s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
sys.path.insert(0,str(RECIPE));import common
ac=load('ac',EXP86/'common.py');reg=copy.deepcopy(common.EXPERIMENTS['vggss_144k']);reg['base_checkpoint']=str(E/'checkpoints/C3/vggss/seed12345/selected_best.pth');cfg=common.load_base_config(reg);cfg.workers=2;dev=torch.device('cuda:0');model,_=common.build_model(cfg,reg,dev);ck=torch.load(E/'stage2/C3/vggss/seed12345/vggss_best.pth',map_location='cpu',weights_only=False);model.student.proj3_spatial.load_state_dict(ck['proj3_spatial_state_dict']);model.student.adapter.load_state_dict(ck['topdown_adapter_state_dict']);model.eval();d=ac.test_dataset('vggss');im,au,gt,names,_=next(iter(DataLoader(Subset(d,range(32)),batch_size=32)));im,au,gt,names=common.flatten_eval_batch(im,au,gt,names);q=torch.from_numpy(np.load(E/'natural_localization/vggss/C3/seed12345/audio_queries.npy')[:32]).to(dev)
with torch.inference_mode():
 t=model._extract_visual_teacher(im.to(dev).float(),q);f=model._fine_from_teacher_features(t,q);branch=model.teacher.slot_attn.visual_branches[-1];slots=model.teacher.slot_attn.slots.expand(len(q),-1,-1);_,vq,_=branch(model._to_tokens(t['f4_projected']),slots)
 def score(x):
  k=branch.img_to_k(branch.img_norm_input(model._to_tokens(x)));a=model.teacher.slot_attn._attention(q,k,model.teacher.infer_sharpening)[:,0];v=model.teacher.slot_attn._attention(vq,k,model.teacher.infer_sharpening)[:,0];return -((a-v)**2).mean(1)
 base=score(f['F34']);x=f['F34'];zero=x.clone();zero[:,:,5:8,5:8]=0;mean=x.clone();mean[:,:,5:8,5:8]=x.mean((2,3),True);out={}
 for n,y in [('zero',zero),('mean',mean)]:out[n]={'norm_rel':float((y-x).norm()/x.norm()),'score_abs_shift':float((score(y)-base).abs().mean()),'score_drop_mean':float((base-score(y)).mean())}
(R/'intervention/intervention_sanity.json').write_text(json.dumps(out,indent=2));print(out)
