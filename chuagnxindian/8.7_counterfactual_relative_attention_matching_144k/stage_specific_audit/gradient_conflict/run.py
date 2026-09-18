"""Short unchanged-25k trajectory and read-only grouped gradient measurements."""
from pathlib import Path
import argparse, csv, importlib.util, json, math, subprocess, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parents[1]
EXP = HERE.parent
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module
T = load('stage_specific_trainer', EXP / 'stage2_with_cram/train.py')

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + '\n')

def norm(values):
    return math.sqrt(sum(float(x.double().square().sum()) for x in values))

def compare(a, b, total):
    na, nb, nt = norm(a), norm(b), norm(total)
    dot = sum(float((x.double()*y.double()).sum()) for x,y in zip(a,b))
    residual = norm([z-x-y for x,y,z in zip(a,b,total)])
    return dict(base_norm=na, weighted_cram_norm=nb, raw_cram_norm=nb/25000,
                total_norm=nt, ratio=nb/na if na else math.nan,
                cosine=dot/(na*nb) if na and nb and math.isfinite(na*nb) else math.nan,
                sum_relative_error=residual/max(nt,1e-30))

def gradients(model, image, audio, geom, wrong, amp, scale):
    with torch.amp.autocast('cuda', enabled=amp):
        out = model.forward_two_views(image, audio, geom)
        base = model.spatial_losses(out)
        cram = model.stage2_cram_components(out, wrong)
        total = base['loss_total'] + 25000*cram['cram_loss']
    named = list(model.student.named_parameters())
    targets = [p for _,p in named] + [out['FINE_KEYS_A'],out['F34_A']]
    grads = []
    for value in (base['loss_total'],25000*cram['cram_loss'],total):
        gg = torch.autograd.grad(scale*value, targets, retain_graph=True, allow_unused=True)
        grads.append([torch.zeros_like(p,dtype=torch.float32) if g is None else g.detach().float()/scale for g,p in zip(gg,targets)])
    groups = {'all_student':list(range(len(named))),
              'proj3_spatial':[i for i,(n,_) in enumerate(named) if n.startswith('proj3_spatial')],
              'adapter':[i for i,(n,_) in enumerate(named) if n.startswith('adapter')],
              'adapter_output_conv':[i for i,(n,_) in enumerate(named) if n.startswith('adapter.layers.2')],
              'K34_activation':[len(named)], 'F34_activation':[len(named)+1]}
    records = [{ 'group':name, **compare(*[[g[i] for i in indices] for g in grads])} for name,indices in groups.items()]
    values = dict(base_loss=float(base['loss_total']),coarse_loss=float(base['loss_coarse']),
                  equiv_loss=float(base['loss_equiv']),cram_loss=float(cram['cram_loss']),
                  d_pos=float(cram['d_pos']),d_neg=float(cram['d_neg']),gap=float(cram['g_match']),
                  anchor_entropy=float(-(cram['visual_anchor']*cram['visual_anchor'].clamp_min(1e-30).log()).sum(1).mean()),
                  scale=scale,amp=amp)
    # Weighted gradient is already computed, preserving its actual AMP derivative.
    return records, values, grads[2][:len(named)], {k:list(out[k].shape) for k in ('AUDIO_QUERY_A','VISUAL_QUERY_A','FINE_KEYS_A','F34_A','AUD_FINE_A')}

def state(model, epoch, step):
    return {'epoch':epoch,'diagnostic_step':step,
            'proj3_spatial_state_dict':model.student.proj3_spatial.state_dict(),
            'topdown_adapter_state_dict':model.student.adapter.state_dict()}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--gpu',type=int,required=True)
    args=p.parse_args(); torch.cuda.set_device(args.gpu); device=torch.device('cuda',args.gpu)
    torch.set_num_threads(4)
    outdir=HERE/'gradient_conflict'/args.dataset
    outdir.mkdir(parents=True,exist_ok=False)
    T.original_common.setup_seed(12345)
    model,config,registry,teacher=T.build_model(args.dataset,device)
    data=T.original_common.get_train_dataset(config,hard_img=config.hard_img,hard_aud=config.hard_aud,rand_aud=config.rand_aud)
    loader=DataLoader(data,batch_size=256,shuffle=True,num_workers=8,pin_memory=True,drop_last=True)
    optimizer=torch.optim.AdamW(model.student.parameters(),lr=5e-5,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda')
    scaler.scale(torch.zeros((),device=device))  # initialize lazy scale tensors before measured-gradient updates
    save_json(outdir/'manifest.json',dict(seed=12345,batches=64,lambda_cram=25000,
        config=vars(config),teacher=str(teacher),teacher_sha256=T.sha256(teacher),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=T.ROOT,text=True).strip(),
        source_hashes={str(q):T.sha256(q) for q in (Path(__file__),EXP/'stage2_with_cram/train.py',EXP/'stage2_with_cram/model.py',HERE/'PROTOCOL.md')},
        parameter_groups={n:list(p.shape) for n,p in model.student.named_parameters()},
        frozen_audio=True,frozen_key_projection=True))
    torch.save(state(model,0,0),outdir/'step000.pth')
    records=[]; steps=[]; mappings=[]; fixed=[]
    start=time.time()
    for step,batch in enumerate(loader):
        if step>=64: break
        image,audio,_,names,_=batch
        image=image.to(device).float();audio=audio.to(device).float()
        geom=T.sample_random_resized_crop(len(image),224,224,device,scale=(.6,1.),ratio=(.9,1.1),flip_probability=.5)
        wrong,indices=T.wrong_audio_batch(audio,names,args.dataset,0,step)
        if step<4:
            fixed.append((image.cpu(),audio.cpu(),{k:v.cpu() for k,v in geom.items()},list(names)))
        mappings.append(dict(step=step,matched_ids=list(names),wrong_ids=[[names[i] for i in row] for row in indices]))
        model.train(); optimizer.zero_grad(set_to_none=True)
        rr,vv,gg,shapes=gradients(model,image,audio,geom,wrong,True,float(scaler.get_scale()))
        for row in rr: records.append(dict(dataset=args.dataset,state='short',step=step,precision='amp',**row))
        steps.append(dict(step=step,**vv))
        if step in (0,16,63):
            fp,_,_,_=gradients(model,image,audio,geom,wrong,False,1.)
            for row in fp: records.append(dict(dataset=args.dataset,state='short',step=step,precision='fp32',**row))
        # Same total derivative and AdamW update as production, via the measured gradient.
        if not all(torch.isfinite(g).all() for g in gg): raise RuntimeError('Diagnostic nonfinite derivative')
        for param,g in zip(model.student.parameters(),gg): param.grad=(g*scaler.get_scale()).to(param.dtype)
        scaler.step(optimizer);scaler.update()
        assert not any(p.grad is not None for p in model.teacher.parameters())
        if step+1 in (16,64): torch.save(state(model,0,step+1),outdir/f'step{step+1:03d}.pth')
        if step%8==0: print(args.dataset,step,vv,'all',rr[0],flush=True)
    torch.save(fixed,outdir/'fixed_training_batches.pth')
    save_json(outdir/'negative_mapping.json',mappings)
    save_json(outdir/'tensor_shapes.json',shapes)
    # Old checkpoints: fixed train batches, no optimizer update.
    sweeps=[]
    for tag,fn in [('old25_epoch1',f'{args.dataset}_best.pth'),('old25_latest','latest.pth')]:
        path=EXP/f'stage2_with_cram/checkpoints/lambda025k/{args.dataset}/seed12345/{fn}'
        ck=torch.load(path,map_location='cpu',weights_only=False)
        model.student.proj3_spatial.load_state_dict(ck['proj3_spatial_state_dict'])
        model.student.adapter.load_state_dict(ck['topdown_adapter_state_dict'])
        save_json(outdir/f'{tag}_source.json',dict(path=str(path),sha256=T.sha256(path),epoch=ck['epoch']))
        for j,(im,au,ge,names) in enumerate(fixed):
            im=im.to(device);au=au.to(device);ge={k:v.to(device) for k,v in ge.items()}
            wrong,_=T.wrong_audio_batch(au,names,args.dataset,0,j)
            rr,vv,_,_=gradients(model,im,au,ge,wrong,False,1.)
            for row in rr:records.append(dict(dataset=args.dataset,state=tag,step=j,precision='fp32',**row))
            if tag=='old25_latest' and j==0:
                for exponent in (16,24,28,32):
                    rs,vs,_,_=gradients(model,im,au,ge,wrong,True,float(2**exponent))
                    sweeps.append(dict(exponent=exponent,**vs,groups=rs))
        print(args.dataset,tag,'audited',flush=True)
    for name,rows in [('per_batch_gradients.csv',records),('training_steps.csv',steps)]:
        with (outdir/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save_json(outdir/'loss_scale_sweep.json',sweeps)
    save_json(outdir/'completion.json',dict(complete=True,seconds=time.time()-start,steps=64))
    model.close()

if __name__=='__main__':main()
