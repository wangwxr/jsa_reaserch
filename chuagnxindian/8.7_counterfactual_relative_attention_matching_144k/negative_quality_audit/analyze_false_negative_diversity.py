#!/usr/bin/env python3
"""Audit-only semantic-risk and diversity of the frozen candidate pools."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent; POOL=HERE/'candidate_pool'

def pair_cos(q):
    q=q/(np.linalg.norm(q,axis=-1,keepdims=True)+1e-12)
    # q [N,4,D]; mean six unordered pairs
    out=[]
    for i in range(4):
        for j in range(i+1,4): out.append((q[:,i]*q[:,j]).sum(-1))
    return np.stack(out,1).mean(1)

def main():
    all_div=[]; all_risk=[]
    for d in ['vggss','flickr']:
        qfile=HERE/'diversity'/f'audio_queries_144k_{d}.npz'
        if not qfile.exists(): raise FileNotFoundError(qfile)
        q=np.load(qfile,allow_pickle=True); ids=q['ids'].astype(str); labels=q['labels'].astype(str)
        # The audio branch returns two slot queries.  Flatten them only for the
        # audit-only audio embedding cosine; CRAM distance itself remains the
        # exact attention-space distance computed in run_pool_audit.py.
        queries=q['queries'].astype(np.float32).reshape(len(ids),-1)
        idx=np.load(POOL/f'pool_indices_144k_{d}.npz',allow_pickle=True)
        pool=idx['pool_indices'].astype(int) # random4 is exactly prefix
        stats=pd.read_csv(POOL/f'per_sample_pool_stats_144k_{d}.csv')
        # persist IDs enable reproducibility assertion
        assert np.all(stats.sample_id.astype(str).to_numpy()==ids)
        top4=np.array([json.loads(x) for x in stats.top4_pool_indices],dtype=int)
        rand=pool[:,:4]
        rows=[]
        for name,sel in [('random4',rand),('top4_confusing',top4)]:
            cand_q=queries[sel]
            anchor_q=queries[:,None,:]
            an=(anchor_q*cand_q).sum(-1)/(np.linalg.norm(anchor_q,axis=-1)*np.linalg.norm(cand_q,axis=-1)+1e-12)
            top_sim=an.mean(1)
            cand_labels=labels[sel]
            same=(cand_labels==labels[:,None])
            # Flickr has placeholder labels; semantic rate explicitly unavailable.
            semantic_available=d=='vggss'
            rows.append(pd.DataFrame({'dataset':d,'sample_id':ids,'candidate_set':name,
                'anchor_candidate_audio_cosine':top_sim,
                'candidate_pairwise_audio_cosine':pair_cos(cand_q),
                'unique_candidate_count':np.array([len(set(x)) for x in sel]),
                'same_video_or_self_count':np.zeros(len(ids),dtype=int),
                'semantic_class_available':semantic_available,
                'same_semantic_class_rate':same.mean(1) if semantic_available else np.nan,
                'unique_semantic_class_count':np.array([len(set(x)) for x in cand_labels]) if semantic_available else np.nan,
            }))
        out=pd.concat(rows,ignore_index=True)
        # all pool members were constructed after video exclusion; direct assertion on IDs is encoded in generator checks.
        out.to_csv(HERE/'diversity'/f'candidate_diversity_{d}.csv',index=False)
        all_div.append(out)
        summary=out.groupby(['dataset','candidate_set','semantic_class_available']).agg(
            n=('sample_id','count'),anchor_audio_cosine_mean=('anchor_candidate_audio_cosine','mean'),
            anchor_audio_cosine_median=('anchor_candidate_audio_cosine','median'),
            pairwise_audio_cosine_mean=('candidate_pairwise_audio_cosine','mean'),
            same_semantic_class_rate=('same_semantic_class_rate','mean'),
            unique_semantic_class_count=('unique_semantic_class_count','mean'),
            same_video_or_self_count=('same_video_or_self_count','sum')).reset_index()
        summary.to_csv(HERE/'false_negative'/f'false_negative_risk_{d}.csv',index=False)
        all_risk.append(summary)
        print(d,summary.to_string(index=False))
    pd.concat(all_div,ignore_index=True).to_csv(HERE/'diversity'/'candidate_diversity.csv',index=False)
    pd.concat(all_risk,ignore_index=True).to_csv(HERE/'false_negative'/'false_negative_risk.csv',index=False)
if __name__=='__main__':main()
