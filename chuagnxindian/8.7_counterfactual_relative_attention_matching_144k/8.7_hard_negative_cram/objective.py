"""Mean-compatible and soft-min negative aggregation for Stage-1 CRAM."""
from __future__ import annotations
import importlib.util, math, os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE=Path(__file__).resolve().parent;SOURCE=HERE.parent
spec=importlib.util.spec_from_file_location('_mean_cram_objective',SOURCE/'objective.py');_source=importlib.util.module_from_spec(spec);spec.loader.exec_module(_source)
attention=_source.attention;forward_components=_source.forward_components;inference_tensors=_source.inference_tensors;_wrong_a2v=_source._wrong_a2v

def aggregation_settings(aggregation=None,tau=None):
    aggregation=aggregation or os.environ.get('CRAM_NEGATIVE_AGGREGATION','mean')
    if aggregation not in ('mean','softmin'):raise ValueError(aggregation)
    tau=float(tau if tau is not None else os.environ.get('CRAM_SOFTMIN_TAU','0'))
    if aggregation=='softmin' and not math.isfinite(tau) or aggregation=='softmin' and tau<=0:raise ValueError('softmin tau must be positive')
    return aggregation,tau

def aggregate_negative(distances,aggregation=None,tau=None):
    aggregation,tau=aggregation_settings(aggregation,tau)
    if aggregation=='mean':
        weights=torch.full_like(distances,1.0/distances.shape[1]);return distances.mean(1),weights
    # Stable smooth minimum: the most dangerous negative has the smallest D.
    logits=-(distances-distances.min(1,keepdim=True).values)/tau
    weights=torch.softmax(logits,dim=1)
    hard=distances.min(1).values-tau*(torch.logsumexp(logits,dim=1)-math.log(distances.shape[1]))
    return hard,weights

def _eval_audio_tokens(model,audio):
    """Exact eval-mode wrong-audio encoder, evaluated once for a source batch."""
    was=model.audnet.training;model.audnet.eval()
    try:return model._audio_tokens(model.audnet(audio))
    finally:model.audnet.train(was)

def _wrong_from_tokens(model,tokens,visual_keys):
    slots=model.slot_attn.slots.expand(len(tokens),-1,-1)
    masked=model.slot_attn._masked(tokens,model.slot_attn.mask_token_aud)
    _slots,query,_keys=model.slot_attn.audio_branch(masked,slots)
    _logits,probabilities=attention(query,visual_keys)
    return probabilities[:,0]

def cram_components(model,output,wrong_spec,aggregation=None,tau=None,source_spec=None,negative_indices=None):
    _batch,negatives=wrong_spec.shape[:2];keys=output['visual_keys'];target=output['v2v_prob'][:,0].detach()
    dpos=F.mse_loss(output['a2v_prob'][:,0],target,reduction='none').mean(1);pieces=[]
    if (source_spec is None)!=(negative_indices is None):raise ValueError('source_spec and negative_indices must be supplied together')
    if source_spec is not None:
        if negative_indices.ndim!=2 or negative_indices.shape!=wrong_spec.shape[:2]:raise ValueError('negative index shape mismatch')
        # Validated independently at B=256: this is bit-identical in D_pos,
        # D_neg, loss and every parameter gradient to K separate eval audnet
        # calls. It only avoids encoding the same batch audio K times.
        tokens=checkpoint(lambda item:_eval_audio_tokens(model,item),source_spec.detach().requires_grad_(True),use_reentrant=True)
        for index in range(negatives):
            selected=tokens[negative_indices[:,index]]
            wrong=checkpoint(lambda item,keys:_wrong_from_tokens(model,item,keys),selected,keys,use_reentrant=True)
            pieces.append(F.mse_loss(wrong,target,reduction='none').mean(1))
    else:
        for index in range(negatives):
            audio=wrong_spec[:,index].detach().requires_grad_(True)
            wrong=checkpoint(lambda item,visual_keys:_wrong_a2v(model,item,visual_keys),audio,keys,use_reentrant=True)
            pieces.append(F.mse_loss(wrong,target,reduction='none').mean(1))
    individual=torch.stack(pieces,1);dneg,weights=aggregate_negative(individual,aggregation,tau);loss=F.softplus(dpos-dneg).mean()
    return {'d_pos':dpos.mean(),'d_neg':dneg.mean(),'d_neg_mean':individual.mean(),'d_neg_min':individual.min(1).values.mean(),'g_match':(dneg-dpos).mean(),'cram_loss':loss,'negative_weights':weights.detach(),'effective_negative_number':(1/(weights.square().sum(1))).detach(),'hardest_weight':weights.max(1).values.detach()}

def total_loss(output,cram_loss,lambda_cram):return _source.total_loss(output,cram_loss,lambda_cram)
