# 8.7 VGG Residual OGL Gap Audit

## Scope and validation

This is a read-only audit of the frozen `C3 Stage-1 -> vanilla Stage-2` checkpoint on all 5,158 VGG-SS test samples. AUD is the saved `AUD_FINE` attention. OGL is exactly the existing evaluation: independently min-max normalise bicubic AUD and ImageNet ResNet18 object-prior maps at 224, form `minmax(0.6 * AUD + 0.4 * prior)`, and threshold at .6. Replayed per-sample AUD and OGL IoUs match the existing cache to `1.11e-16` maximum absolute error. No checkpoint, model, dataset split, OGL weight, or prior result was changed.

Groups were fixed before analysis:

| group | condition | samples | fraction | mean delta IoU [95% bootstrap CI] | median |
|---|---|---:|---:|---:|---:|
| A: OGL-help | `OGL - AUD > .10` | 428 | 8.30% | +.1438 [.1400, .1476] | +.1303 |
| B: close | `abs(OGL - AUD) <= .05` | 3,099 | 60.08% | +.0029 [.0020, .0038] | +.0045 |
| transition | remaining samples | 1,374 | 26.64% | +.0145 | +.0558 |
| C: AUD-better | `OGL - AUD < -.10` | 257 | 4.98% | -.1493 [-.1554, -.1435] | -.1339 |

At the official per-image cIoU decision (`IoU >= .5`), OGL has 93 net more successful samples than AUD: 154 net from A, +9 from B, +23 from transition, and -93 from C. A supplies 38.9% of all positive continuous IoU change and 154 / 321 = 48.0% of all OGL-upgraded cIoU decisions, despite only 8.3% of images. Thus the aggregate gap is not a diffuse tiny effect; a compact, diagnosable subgroup is material.

## What OGL changes in its helpful subgroup

All values below are paired OGL-minus-AUD changes for group A. Pixel counts are at 224x224 and compare OGL directly with AUD, not with 1.3G.

| metric | AUD mean | OGL mean | paired change mean [95% CI] | median change |
|---|---:|---:|---:|---:|
| GT response | .7974 | .7881 | -.0093 [-.0142, -.0041] | -.0232 |
| NearFP response | .7339 | .6742 | **-.0597 [-.0643, -.0546]** | -.0680 |
| GT-NearFP gap | .0649 | .1148 | **+.0499 [.0464, .0533]** | +.0457 |
| GT/NearFP AUROC | .6876 | .7578 | **+.0702 [.0620, .0789]** | +.0574 |
| coverage | .8608 | .8841 | +.0233 [.0135, .0342] | -.0021 |
| precision | .4652 | .6143 | **+.1491 [.1428, .1553]** | +.1482 |
| activated area | .4552 | .3475 | **-.1077 [-.1182, -.0981]** | -.1211 |
| normalized entropy | .9914 | .9883 | -.0031 [-.0033, -.0029] | -.0032 |
| RemoveFP | — | — | **6,184.6 [5,818.3, 6,564.9]** | 5,905.5 |
| AddTP | — | — | 868.3 [706.7, 1,031.6] | 13.5 |
| RemoveTP | — | — | 354.8 [305.8, 400.7] | 165.5 |
| AddFP | — | — | 268.9 [215.4, 329.0] | 16.5 |

This is **Case A (RemoveFP/context suppression) with a small complementary AddTP contribution**, not a coverage-deficit-only case. The dominant pattern is that OGL removes an extensive non-GT activation field while maintaining, and on average slightly improving, coverage. The GT response itself decreases slightly, so the benefit cannot be described as simply raising target score. It is a stronger target-versus-context separation at the evaluated threshold.

Correlations with `delta_iou = IoU_OGL - IoU_AUD` support the same conclusion: Spearman is +.495 for RemoveFP, -.521 for RemoveTP, +.470 for OGL coverage, and +.293 for OGL GT-NearFP gap. AddTP is weak (+.031). The OGL gain is therefore primarily associated with removing irrelevant activation *without* removing target activation. It is not explained by object count (Spearman -.011).

## A/B/C sample types

| group | AUD IoU -> OGL IoU | AUD gap -> OGL gap | AUD coverage -> OGL coverage | AUD precision -> OGL precision | mean GT area | very-large GT |
|---|---|---|---|---|---:|---:|
| A OGL-help | .4106 -> .5544 | .0649 -> .1148 | .8608 -> .8841 | .4652 -> .6143 | .2583 | 27.8% |
| B close | .4021 -> .4050 | .0404 -> .0494 | .8113 -> .7789 | .5068 -> .5279 | .2886 | 40.9% |
| C AUD-better | .6027 -> .4534 | .0415 -> .0231 | .7433 -> .5576 | .7924 -> .7678 | .5429 | 81.7% |

The residual OGL-help cases are generally **large rather than small**: 95.6% are labelled large/very-large by the pre-existing baseline metadata. They are also disproportionately baseline `OVER` taxonomy (67.1% vs 54.6% in B; only 17.9% in C). This is consistent with over-activation/context confusion, not missing small-object completeness. Group C is dominated by very-large, already-high-IoU, high-precision targets; the object prior then contracts maps too aggressively and removes target pixels (mean RemoveTP 5,159.9).

Native-map normalized entropy is near maximal in every group (AUD: A .9914,
B .9899, C .9921), and connected-component count is near one (A 1.04, B
1.14, C 1.07). Thus this audit does not support a separate diffuse-versus-
concentrated or multi-object explanation for the residual; the decisive
difference is the thresholded context activation and its removability.

There is no dense semantic-context annotation for this full test set. Existing manual `spatial_error` labels cover only 9/428 group-A samples, so they are reported in `metadata_composition.csv` but are not used for a semantic claim. GT connected-component count is also essentially uninformative here.

Top OGL-help examples are saved in `top_ogl_help_samples.csv`; for example, `XHukxF8iWE0_000400` changes .2871 -> .6457 by removing 15,613 FP pixels while removing 146 TP pixels. The strongest AUD-better examples are saved separately; `MQtUV_cFcm8_000020` changes .7844 -> .3895 because OGL removes 14,432 TP pixels. These are identifiers for inspection, not hand-selected evidence for the aggregate conclusion.

## Decision for CRAM

The residual does **not** establish a separate orthogonal completeness failure mode. On VGG, the clearest remaining gap is target/context suppression and hard spatial discrimination. That is still naturally within CRAM's stated problem, although the current uniform mean-over-four-negative CRAM does not fully remove it.

Do not start a second innovation. Do not retry Stage-2 CRAM.

The single most justified next method experiment is a **Stage-1-only, hard-negative-aware CRAM aggregation**, with the same structure on VGG and Flickr: keep all existing losses, positive term, K=4 fixed wrong audios, and Stage-2 vanilla; replace only the uniform `mean_k D_neg,k` in CRAM by a pre-registered smooth **soft-min** over the four distances (for example `-tau*logmeanexp(-D_neg/tau)`, with fixed temperature). The hard wrong audio here is the one with the *smallest* distance, i.e. the one that can best reproduce the visual spatial template. This focuses learning on that dangerous shortcut without adding an object branch, OGL, mask, or third stage.

Before a full training run, perform a training-batch gradient/loss audit and one controlled Stage-1 run. Hold `lambda=1.0`, K=4, seed, batch order and all Stage-2 settings fixed. The current C1/C2/C3 evidence makes blind lambda-only search weakly justified: C3 already had the strongest tested Stage-1 VGG cIoU/coverage among lambda .1/.5/1.0, while this residual audit identifies a *which-negative* limitation rather than global underweighting. If a second scalar test is needed after that, change only the hard-negative temperature; do not combine it with lambda/warmup/negative-count changes.

Flickr already has higher final AUD than OGL because its AUD has materially stronger final GT/NearFP separation (.0981 gap, .7078 AUROC) and high precision (.7256), while OGL's object-prior contraction lowers its cIoU (.8960 -> .8880). VGG remains below OGL because its final AUD gap (.0455) and precision (.5376) leave a substantial over-activation/context subgroup for the prior to remove. There is a realistic, but unproven, route for a CRAM-family improvement to close the VGG gap: 93 cIoU decisions separate current AUD from OGL, and the largest positive subgroup is aligned with the original discrimination objective. This audit is evidence for one targeted CRAM refinement, not a guarantee that it will exceed OGL or a basis for test-set tuning.
