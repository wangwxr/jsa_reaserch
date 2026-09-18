"""Aggregate measured diagnostics, paired intervals, and standalone plots."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parents[1];EXP=HERE.parent
sys.path.insert(0,str(HERE/'gradient_conflict'))
from run import load,T
A=load('stage_specific_summary_metrics',EXP/'mechanism_audit/scripts/analyze.py')
OUT=HERE/'summary';OUT.mkdir(exist_ok=True)

def table(df):return df.to_markdown(index=False,floatfmt='.4f')

def main():
    stats=[];allgrad=[];mechanism=[];perf=[];paired=[];components=[];numerical={};checks={}
    for ds in ('vggss','flickr'):
        g=HERE/'gradient_conflict'/ds;d=HERE/'degradation_path'/ds
        frame=pd.read_csv(g/'per_batch_gradients.csv');allgrad.append(frame)
        for (state,precision,group),q in frame.groupby(['state','precision','group']):
            c=q.cosine.dropna();row=dict(dataset=ds,state=state,precision=precision,group=group,n=len(q),valid_cosines=len(c))
            for metric in ('base_norm','weighted_cram_norm','raw_cram_norm','total_norm','ratio','cosine','sum_relative_error'):
                vals=q[metric].replace([np.inf,-np.inf],np.nan).dropna()
                row.update({f'{metric}_mean':vals.mean(),f'{metric}_median':vals.median(),f'{metric}_p25':vals.quantile(.25),f'{metric}_p75':vals.quantile(.75)})
            row.update(negative_fraction=float((c<0).mean()),below_minus025=float((c<-.25).mean()),below_minus05=float((c<-.5).mean()))
            stats.append(row)
        mechanism.append(pd.read_csv(d/'mechanism_summary.csv'))
        perf.append(pd.read_csv(d/'six_readout_summary.csv'))
        components.append(pd.read_csv(g/'component_gradients.csv'))
        numerical[ds]=json.load(open(g/'loss_scale_sweep.json'))
        checks[ds]=dict(gradients=json.load(open(g/'completion.json')),evaluation=json.load(open(d/'validation.json')),parity=json.load(open(g/'wrapper_parity.json')))
        samples=pd.read_csv(d/'six_readout_per_sample.csv',dtype={'sample_id':str})
        for candidate,reference in [('cram_c3_stage2','original_1.3g_final'),('lambda025k_best','cram_c3_stage2'),('lambda025k_latest','cram_c3_stage2')]:
            c=samples[samples.model.eq(candidate)].sort_values('sample_id');r=samples[samples.model.eq(reference)].sort_values('sample_id')
            assert np.array_equal(c.sample_id,r.sample_id)
            for readout in ('AUD','OGL'):
                for metric in ('cIoU','AUC'):
                    x=c[readout].to_numpy();y=r[readout].to_numpy()
                    if metric=='cIoU':diff=(x>=.5).astype(float)-(y>=.5).astype(float)
                    else:
                        t=np.arange(21)*.05
                        diff=np.trapezoid((x[:,None]>=t).astype(float),t,axis=1)-np.trapezoid((y[:,None]>=t).astype(float),t,axis=1)
                    paired.append(dict(dataset=ds,candidate=candidate,reference=reference,readout=readout,metric=metric,**A.stats(diff,ds,candidate,reference,readout,metric)))
    gs=pd.DataFrame(stats);gd=pd.concat(allgrad);ms=pd.concat(mechanism);ps=pd.concat(perf);ci=pd.DataFrame(paired);cs=pd.concat(components)
    gs.to_csv(OUT/'gradient_group_summary.csv',index=False);gd.to_csv(OUT/'all_batch_gradients.csv',index=False)
    ms.to_csv(OUT/'stage_specific_summary.csv',index=False);ps.to_csv(OUT/'six_readout_performance.csv',index=False);ci.to_csv(OUT/'performance_paired_CI.csv',index=False)
    cs.groupby(['dataset','state','term'])[['cosine','base_norm','weighted_cram_norm']].mean().to_csv(OUT/'component_summary.csv')
    payload=dict(mechanism=ms.to_dict('records'),performance=ps.to_dict('records'),gradient_statistics=gs.to_dict('records'),paired_performance_ci=ci.to_dict('records'),checks=checks,
        conclusion='Conflict emerges early in shared refinement parameters; Flickr loses coverage despite stronger GT/NearFP separation; VGG loses both. AMP scaling susceptibility is distinct from objective conflict. Keep Stage-1 CRAM plus vanilla Stage-2 as main method.',
        missing_checkpoints='No epoch5/10/20 or final states; old epoch1/latest are explicitly labeled.',
        numerical_audit=numerical)
    (OUT/'stage_specific_summary.json').write_text(json.dumps(payload,indent=2,default=str)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for col,ds in enumerate(('vggss','flickr')):
        for group in ('all_student','proj3_spatial','adapter','K34_activation','F34_activation'):
            q=gd[(gd.dataset==ds)&(gd.state=='short')&(gd.precision=='amp')&(gd.group==group)]
            axes[0,col].plot(q.step,q.cosine,label=group,lw=1.25)
        q=gd[(gd.dataset==ds)&(gd.state=='short')&(gd.precision=='amp')&(gd.group=='all_student')]
        for name in ('base_norm','weighted_cram_norm','total_norm'):axes[1,col].plot(q.step,q[name],label=name)
        axes[0,col].axhline(0,c='black',lw=.7);axes[0,col].set_ylim(-1.05,1.05);axes[0,col].set_title(ds)
        axes[0,col].set_ylabel('Gradient cosine');axes[1,col].set_ylabel('L2 norm');axes[1,col].set_yscale('log');axes[1,col].set_xlabel('Diagnostic update (0-based)')
        axes[0,col].legend(fontsize=7);axes[1,col].legend(fontsize=7)
    for suffix in ('png','pdf'):fig.savefig(OUT/f'gradient_conflict_trends.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    models=['short_step000','short_step064','lambda025k_best','lambda025k_latest']
    for row,ds in enumerate(('vggss','flickr')):
        q=ms[ms.dataset.eq(ds)].set_index('model').loc[models]
        for col,names in enumerate([('gt_nearfp_gap','gt_vs_nearfp_auroc'),('coverage','precision','pred_area_ratio'),('cIoU','AUC')]):
            for name in names:axes[row,col].plot(range(4),q[name],marker='o',label=name)
            axes[row,col].set_xticks(range(4),['init','step64*','old ep1','old late'],rotation=15)
            axes[row,col].set_title(ds);axes[row,col].legend(fontsize=8)
    fig.suptitle('25k degradation: *short diagnostic and old run are different trajectories')
    for suffix in ('png','pdf'):fig.savefig(OUT/f'degradation_path.{suffix}',dpi=180)
    plt.close(fig)
    # Data appendix is regenerated; interpretation lives in the main report.
    lines=['# Measured data appendix','', 'Gradient cosines: undefined zero-gradient groups excluded from fractions.','']
    pick=gs[(gs.precision=='fp32')&(gs.state!='short')]
    lines.extend([table(pick[['dataset','state','group','cosine_mean','cosine_median','cosine_p25','cosine_p75','negative_fraction','below_minus025','below_minus05','ratio_mean']]),''])
    for ds in ('vggss','flickr'):
        lines.extend([f'## {ds}', '',table(ms[ms.dataset==ds][['model','gt_response','nearfp_response','gt_nearfp_gap','gt_vs_nearfp_auroc','coverage','precision','pred_area_ratio','RemoveFP','AddTP','RemoveTP','AddFP','normalized_entropy','cIoU','AUC']]),'',table(ps[(ps.dataset==ds)&ps.readout.isin(['AUD','OGL'])]),''])
    lines.extend(['## Paired per-image performance intervals','',table(ci[['dataset','candidate','reference','readout','metric','mean','ci_low','ci_high']])])
    (OUT/'measured_data.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
