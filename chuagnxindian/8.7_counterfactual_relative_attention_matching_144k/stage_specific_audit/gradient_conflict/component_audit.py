"""Read-only FP32 decomposition, wrapper parity and local directional intervention."""
from run import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--gpu',type=int,required=True);args=p.parse_args()
    torch.set_num_threads(4);torch.cuda.set_device(args.gpu);device=torch.device('cuda',args.gpu)
    outdir=HERE/'gradient_conflict'/args.dataset
    model,_,_,_=T.build_model(args.dataset,device)
    fixed=torch.load(outdir/'fixed_training_batches.pth',weights_only=False)
    paths={'initial':outdir/'step000.pth','epoch1':EXP/f'stage2_with_cram/checkpoints/lambda025k/{args.dataset}/seed12345/{args.dataset}_best.pth',
        'late':EXP/f'stage2_with_cram/checkpoints/lambda025k/{args.dataset}/seed12345/latest.pth'}
    records=[];trials=[];parities=[]
    for tag,path in paths.items():
        ck=torch.load(path,map_location=device,weights_only=False)
        model.student.proj3_spatial.load_state_dict(ck['proj3_spatial_state_dict']);model.student.adapter.load_state_dict(ck['topdown_adapter_state_dict'])
        for ix,(im,au,ge,names) in enumerate(fixed):
            im=im.to(device);au=au.to(device);ge={k:v.to(device) for k,v in ge.items()}
            wrong,_=T.wrong_audio_batch(au,names,args.dataset,0,ix)
            model.train();out=model.forward_two_views(im,au,ge);base=model.spatial_losses(out);cram=model.stage2_cram_components(out,wrong)
            params=list(model.student.parameters())
            terms={'base':base['loss_total'],'coarse':base['loss_coarse'],'equiv':base['loss_equiv'],
                   'cram_weighted':25000*cram['cram_loss'],'positive_half':12500*cram['d_pos'],'negative_half':-12500*cram['d_neg']}
            grads={k:[torch.zeros_like(p) if g is None else g.detach() for g,p in zip(torch.autograd.grad(v,params,retain_graph=True,allow_unused=True),params)] for k,v in terms.items()}
            # Activation K34_A captures view A only, whereas parameter gradients
            # sum both views. Separate these before interpreting their cosines.
            keygrad=torch.autograd.grad(base['loss_total'],out['FINE_KEYS_A'],retain_graph=True)[0]
            ga=torch.autograd.grad(out['FINE_KEYS_A'],params,grad_outputs=keygrad.detach(),retain_graph=True,allow_unused=True)
            grads['base_view_a']=[torch.zeros_like(p) if g is None else g.detach() for g,p in zip(ga,params)]
            grads['base_view_b']=[g-a for g,a in zip(grads['base'],grads['base_view_a'])]
            terms['base_view_a']=base['loss_total'];terms['base_view_b']=base['loss_total']
            for k in ('base','coarse','equiv','positive_half','negative_half','base_view_a','base_view_b'):
                a=grads[k];b=grads['cram_weighted'];r=compare(a,b,[x+y for x,y in zip(a,b)])
                records.append(dict(dataset=args.dataset,state=tag,batch=ix,term=k,loss=float(terms[k]),**r))
            if ix==0:
                # Base class and wrapper share the same tensors/parameters; verify inherited behavior.
                with torch.no_grad():
                    original=T.Stage2WithCRAM.__mro__[1].forward_two_views(model,im,au,ge)
                    parities.append(dict(state=tag,max_output_error=max(float((original[k]-out[k]).abs().max()) for k in ('AUD_FINE_A','AUD_FINE_B','AUD_FINE_B_TO_A','AUD_L4_A')),
                                         original_loss_error=abs(float(model.spatial_losses(original)['loss_total'])-float(base['loss_total']))))
                saved=[p.detach().clone() for p in params]
                initial_base=float(base['loss_total']);initial_pos=float(cram['d_pos']);initial_neg=float(cram['d_neg'])
                for direction in ('base','cram_weighted'):
                    # A diagnostic local step, restored immediately; no continuation/checkpoint selection.
                    length=1e-3;gn=norm(grads[direction])
                    with torch.no_grad():
                        for p0,v,g in zip(params,saved,grads[direction]):p0.copy_(v-length*g/max(gn,1e-30))
                        oo=model.forward_two_views(im,au,ge);bb=model.spatial_losses(oo);cc=model.stage2_cram_components(oo,wrong)
                        trials.append(dict(state=tag,direction=direction,parameter_step_l2=length,base_before=initial_base,
                            base_after=float(bb['loss_total']),base_delta=float(bb['loss_total'])-initial_base,
                            dpos_before=initial_pos,dpos_after=float(cc['d_pos']),dneg_before=initial_neg,dneg_after=float(cc['d_neg'])))
                        for p0,v in zip(params,saved):p0.copy_(v)
    with (outdir/'component_gradients.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    save_json(outdir/'local_directional_checks.json',trials)
    save_json(outdir/'wrapper_parity.json',parities)
    assert max(p['max_output_error'] for p in parities)==0
    model.close();print(args.dataset,'component audit complete',flush=True)

if __name__=='__main__':main()
