#!/usr/bin/env python3
"""Reproducible publication-style figures for the frozen cross-scale audit."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9,'axes.titlesize':10,'axes.titleweight':'bold','axes.labelsize':9,'legend.fontsize':8,'legend.frameon':False,'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16})
C = {'baseline':'#B0BEC5','mean':'#E76F51','hard_default':'#0072B2','hard_adjusted':'#56B4E9'}

def save(fig, name):
    fig.savefig(HERE / f'{name}.pdf')
    fig.savefig(HERE / f'{name}.png', dpi=300)
    plt.close(fig)

scale = pd.read_csv(ROOT / 'scale_comparison/10k_vs_144k.csv')
fig, axs = plt.subplots(1, 2, figsize=(6.75, 2.45), sharey=False)
for ax, metric, title in zip(axs, ['hard_effect_cIoU','hard_effect_AUC'], ['Hard-default effect on cIoU','Hard-default effect on AUC']):
    x = np.arange(2); width = .34
    for j, ds in enumerate(['vggss','flickr']):
        z = scale[scale.dataset.eq(ds)].set_index('scale').loc[['10k','144k'],metric].to_numpy()
        ax.bar(x + (j-.5)*width, z, width, label='VGG-SS' if ds=='vggss' else 'Flickr', color=['#0072B2','#2A9D8F'][j])
    ax.axhline(0,color='#444',lw=.8); ax.set_xticks(x,['10k','144k']); ax.set_title(title); ax.set_xlabel('Stage-1 training set size'); ax.set_ylabel('Hard-default − Mean'); ax.legend()
save(fig, 'hard_effect_10k_vs_144k')

ga = pd.read_csv(ROOT / 'residual_group/groupA_comparison.csv').set_index('model')
fig, axs = plt.subplots(1, 3, figsize=(6.75, 2.45))
for ax, col, title in zip(axs, ['iou','precision','coverage'], ['AUD IoU','Precision','Coverage']):
    methods = ['mean','hard_default','hard_adjusted']
    ax.bar(np.arange(3), [ga.loc[m,col] for m in methods], color=[C[m] for m in methods])
    ax.set_xticks(np.arange(3), ['Mean','Hard\ndef.','Hard\nadj.']); ax.set_title(f'Group-A: {title}'); ax.set_ylim(0,1)
save(fig, 'groupA_vs_global')

h = pd.read_csv(ROOT / 'hardness_signal/per_sample_signals.csv')
h = h[(h.scale=='144k') & (h.dataset=='vggss')].copy(); h['subset'] = np.where(h.group.eq('A_ogl_help'),'OGL-help Group A','All other')
fig, axs = plt.subplots(1, 3, figsize=(6.75, 2.45))
for ax, col, title in zip(axs, ['mean_minus_min','weight_max','N_eff'], ['mean − min distance','hardest weight','effective negatives']):
    vals = [h[h.subset.eq(s)][col].dropna().to_numpy() for s in ['All other','OGL-help Group A']]
    bp = ax.boxplot(vals,patch_artist=True,showfliers=False,widths=.6)
    for box,color in zip(bp['boxes'],['#B0BEC5','#E76F51']): box.set_facecolor(color)
    ax.set_xticks([1,2],['Other','Group A']); ax.set_title(title)
save(fig, 'hardness_signal_distributions')

hh = pd.read_csv(ROOT / 'residual_group/hard_help_hurt.csv')
fig, axs = plt.subplots(1, 3, figsize=(6.75, 2.45))
for ax, col, title in zip(axs, ['mean_iou','precision','coverage'], ['Mean AUD IoU','Mean precision','Mean coverage']):
    z = hh[hh.dataset.eq('vggss')].set_index('subset'); order=['hard_help','hard_hurt']
    ax.bar(np.arange(2),[z.loc[o,col] for o in order],color=['#2A9D8F','#D55E00'])
    ax.set_xticks([0,1],['Hard-help','Hard-hurt']); ax.set_ylim(0,1); ax.set_title(title)
save(fig, 'help_hurt_comparison')

if __name__=='__main__':
    print('saved figures')
