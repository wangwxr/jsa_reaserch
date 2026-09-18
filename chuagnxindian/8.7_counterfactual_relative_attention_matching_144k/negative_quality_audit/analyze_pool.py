#!/usr/bin/env python3
"""Aggregate the read-only M=32 negative-quality audit.

No training, checkpoint, loss, or candidate-selection logic is changed here.
GT/OGL/group labels are loaded only after pool statistics have been frozen, for
analysis of their association with the internally computed distances.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
POOL = HERE / "candidate_pool"
NBOOT = 4000
SEED = 20260916

MEASURES = [
    "opportunity_gap_mean", "opportunity_gap_min", "violation_fraction_pool",
    "random4_violation_fraction", "missed_confusing_negative", "pool_min",
    "top4_mean", "top4_vs_random4_ratio", "pool_std",
]

def rng_for(key: str):
    import hashlib
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))

def clean(x):
    return pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)

def describe(x, label=""):
    x = clean(x)
    if not len(x): return {"n": 0}
    r = rng_for("describe:" + label)
    means = np.empty(NBOOT)
    for i in range(NBOOT): means[i] = x[r.integers(0, len(x), len(x))].mean()
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
            "p25": float(np.quantile(x,.25)), "p75": float(np.quantile(x,.75)),
            "p90": float(np.quantile(x,.90)), "p95": float(np.quantile(x,.95)),
            "bootstrap_mean_ci_low": float(np.quantile(means,.025)),
            "bootstrap_mean_ci_high": float(np.quantile(means,.975))}

def compare(a, b, label=""):
    a, b = clean(a), clean(b)
    out = {"left": describe(a,label+":l"), "right": describe(b,label+":r")}
    if not len(a) or not len(b): return out
    r = rng_for("compare:" + label)
    diffs = np.empty(NBOOT)
    for i in range(NBOOT):
        diffs[i] = a[r.integers(0,len(a),len(a))].mean() - b[r.integers(0,len(b),len(b))].mean()
    u,p = mannwhitneyu(a,b,alternative="two-sided")
    # rank-biserial effect: positive means left tends to be larger.
    rb = 2*u/(len(a)*len(b))-1
    out.update({"mean_diff_left_minus_right": float(a.mean()-b.mean()),
                "bootstrap_ci_low": float(np.quantile(diffs,.025)),
                "bootstrap_ci_high": float(np.quantile(diffs,.975)),
                "mannwhitney_p": float(p), "rank_biserial": float(rb)})
    return out

def flatten(title, comp):
    rows=[]
    for measure, d in comp.items():
        row={"comparison":title,"measure":measure}
        for side in ("left","right"):
            for k,v in d[side].items(): row[f"{side}_{k}"]=v
        for k in ("mean_diff_left_minus_right","bootstrap_ci_low","bootstrap_ci_high","mannwhitney_p","rank_biserial"):
            row[k]=d.get(k,np.nan)
        rows.append(row)
    return rows

def load_pool(scale,dataset):
    return pd.read_csv(POOL/f"per_sample_pool_stats_{scale}_{dataset}.csv")

def main():
    for p in [HERE/'groupA',HERE/'ogl_relation',HERE/'hard_help_hurt',HERE/'false_negative',HERE/'diversity',HERE/'cross_scale',HERE/'summary',HERE/'figures']:
        p.mkdir(parents=True,exist_ok=True)
    datasets=[('144k','vggss'),('144k','flickr'),('10k','vggss'),('10k','flickr')]
    pools={(s,d):load_pool(s,d) for s,d in datasets}
    # Global pool coverage controls across scale / dataset.
    global_rows=[]
    for (s,d),x in pools.items():
        for m in ["d_pos","current_random4_mean","current_random4_min","pool_min","pool_mean","pool_std","top4_mean","opportunity_gap_mean","opportunity_gap_min","violation_fraction_pool","random4_violation_fraction","missed_confusing_negative"]:
            z=describe(x[m],f"global:{s}:{d}:{m}")
            global_rows.append({"scale":s,"dataset":d,"metric":m,**z})
    pd.DataFrame(global_rows).to_csv(HERE/'cross_scale/cross_scale_summary.csv',index=False)
    # fixed VGG residual groups and continuous residual covariates
    residual=pd.read_csv(EXP/'residual_ogl_gap_audit/per_sample/vggss_aud_vs_ogl_per_sample.csv')
    vgg=pools[('144k','vggss')].merge(residual,on=['sample_id','sample_index'],how='left',validate='one_to_one')
    assert vgg['group'].notna().all(), 'fixed Group-A mapping failed'
    groupa=vgg.group.eq('A_ogl_help')
    gcomp={m:compare(vgg.loc[groupa,m],vgg.loc[~groupa,m],f'groupA:{m}') for m in MEASURES}
    pd.DataFrame(flatten('Group-A vs non-Group-A',gcomp)).to_csv(HERE/'groupA/groupA_vs_non_groupA.csv',index=False)
    # OGL-help vs AUD-better fixed groups (not a replacement for Group-A)
    ogl=vgg[vgg.group.isin(['A_ogl_help','C_aud_better'])].copy()
    ocomp={m:compare(ogl.loc[ogl.group.eq('A_ogl_help'),m],ogl.loc[ogl.group.eq('C_aud_better'),m],f'ogl:{m}') for m in MEASURES}
    pd.DataFrame(flatten('OGL-help vs AUD-better',ocomp)).to_csv(HERE/'ogl_relation/ogl_help_vs_aud_better.csv',index=False)
    # Correlation with the frozen residual diagnostics. Again all post-hoc analysis only.
    corr_rows=[]
    covars=['delta_iou','aud_precision','aud_coverage','aud_area','aud_gap','aud_nearfp_response','gt_area_ratio']
    for q in ['opportunity_gap_mean','opportunity_gap_min','violation_fraction_pool','missed_confusing_negative','pool_min','top4_mean']:
        for c in covars:
            xx=pd.to_numeric(vgg[q],errors='coerce'); yy=pd.to_numeric(vgg[c],errors='coerce'); keep=xx.notna()&yy.notna()
            rho,p=spearmanr(xx[keep],yy[keep])
            corr_rows.append({'quality_metric':q,'residual_metric':c,'n':int(keep.sum()),'spearman_rho':float(rho),'p_value':float(p)})
    # taxonomy relation uses AUC-less grouping summary, not candidate selection
    tax_rows=[]
    for tax,sub in vgg.groupby('auto_taxonomy',dropna=False):
        for q in ['opportunity_gap_mean','missed_confusing_negative','violation_fraction_pool','pool_min']:
            tax_rows.append({'taxonomy':str(tax),'metric':q,**describe(sub[q],f'tax:{tax}:{q}')})
    pd.DataFrame(corr_rows).to_csv(HERE/'ogl_relation/correlations.csv',index=False)
    pd.DataFrame(tax_rows).to_csv(HERE/'ogl_relation/over_taxonomy_quality.csv',index=False)
    # Hard help/hurt—definition frozen in cross-scale audit.
    hh_rows=[]
    for d in ['vggss','flickr']:
        hp=pd.read_csv(EXP/f'cross_scale_cram_audit/residual_group/hard_help_hurt_per_sample_{d}.csv')
        x=pools[('144k',d)].merge(hp[['sample_id','subset']],on='sample_id',how='left',validate='one_to_one')
        for m in MEASURES:
            z=compare(x.loc[x.subset.eq('hard_help'),m],x.loc[x.subset.eq('hard_hurt'),m],f'hh:{d}:{m}')
            hh_rows.extend(flatten(f'{d}: Hard-help vs Hard-hurt',{m:z}))
    pd.DataFrame(hh_rows).to_csv(HERE/'hard_help_hurt/hard_help_vs_hurt.csv',index=False)
    # Preserve a convenient analysis-only merged VGG file (pool values + frozen residual labels).
    vgg.to_csv(HERE/'groupA/vggss_144k_pool_with_fixed_residual_labels.csv',index=False)
    # Machine-readable audit bundle for prose generation.
    bundle={
        'protocol': {'read_only':True,'M':32,'random4':'existing Experiment 8.1 set1 prefix; uniform different-video test-side surrogate',
                     'selection_features':'only D_neg from frozen Mean Stage-1; no GT/OGL/class/group label in selection'},
        'global': global_rows,
        'groupA_vs_non_groupA':gcomp,
        'ogl_help_vs_aud_better':ocomp,
    }
    (HERE/'summary/analysis_bundle.json').write_text(json.dumps(bundle,indent=2))
    print('wrote aggregate CSVs and analysis bundle')

if __name__=='__main__': main()
