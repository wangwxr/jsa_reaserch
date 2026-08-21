# Experiment 5.0 - Agreement-Disagreement Pseudo-Label Purity Probe

## Protocol Audit

- Zero training: `model.eval()`, `torch.inference_mode()`, no optimizer, no backward, and zero trainable parameters.
- Primary teacher is frozen Stage1 `AUD_L4` / `IMG_L4` at `7x7`. Stage2 `AUD_FINE` / aligned `IMG` at `14x14` is diagnostic only.
- Formal seed is fixed Top20: Stage1 `k=10`, Stage2 `k=40`; stable descending flat-index tie breaking; binary seeds are nearest-neighbor resized to the binary `224x224` GT only for analysis.
- Empty seeds are excluded from purity means and retained in non-empty/empty-rate reporting. GT and OGL are never model or pseudo-label inputs.

## Reproduction And Shapes

### VGGSS-144k

- Stage1 AUD/IMG: `0.4002/0.4127` / `0.4069/0.4166`; Stage2 AUD_FINE/IMG: `0.4269/0.4230` / `0.4069/0.4166`.
- Raw map max error vs 4.1: `{'AUD_L4': 0.0, 'IMG_L4': 0.0, 'AUD_FINE': 0.0}`; per-sample metric max error vs 4.0/4.1: `{'Stage1_AUD': 0.0, 'Stage1_IMG': 0.0, 'Stage2_AUD': 0.0, 'Stage2_IMG': 0.0}`; vs 4.2: `{'Stage2_AUD': 0.0, 'Stage2_IMG': 0.0}`; sample mismatches: `0`.
- Shapes: `Qa=[256, 2, 512]`, `Qv=[256, 2, 512]`, `K4=[256, 49, 512]`, `K34=[256, 196, 512]`, `AUD_L4=[256, 1, 7, 7]`, `IMG_L4=[256, 1, 7, 7]`, `AUD_FINE=[256, 1, 14, 14]`, aligned IMG=`[256, 1, 14, 14]`, GT=`[256, 224, 224]`.
- Checkpoints unchanged: `True`; no NaN/Inf: `True`; trainable parameters: `0`.

### Flickr-144k

- Stage1 AUD/IMG: `0.8040/0.6228` / `0.8040/0.6166`; Stage2 AUD_FINE/IMG: `0.8120/0.6356` / `0.8040/0.6166`.
- Raw map max error vs 4.1: `{'AUD_L4': 0.0, 'IMG_L4': 0.0, 'AUD_FINE': 0.0}`; per-sample metric max error vs 4.0/4.1: `{'Stage1_AUD': 0.0, 'Stage1_IMG': 0.0, 'Stage2_AUD': 0.0, 'Stage2_IMG': 0.0}`; vs 4.2: `{'Stage2_AUD': 0.0, 'Stage2_IMG': 0.0}`; sample mismatches: `0`.
- Shapes: `Qa=[32, 2, 512]`, `Qv=[32, 2, 512]`, `K4=[32, 49, 512]`, `K34=[32, 196, 512]`, `AUD_L4=[32, 1, 7, 7]`, `IMG_L4=[32, 1, 7, 7]`, `AUD_FINE=[32, 1, 14, 14]`, aligned IMG=`[32, 1, 14, 14]`, GT=`[32, 224, 224]`.
- Checkpoints unchanged: `True`; no NaN/Inf: `True`; trainable parameters: `0`.

## Stage1 Primary Purity

Macro is the primary column; micro pools all seed pixels.

| Dataset | Seed | Macro FG | Macro BG | Micro FG | Micro BG | FG recall | BG coverage | Non-empty | Mean area |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VGGSS-144k | Random A-matched | 0.3136 | 0.6864 | 0.3136 | 0.6864 | 0.2065 | 0.2042 | 1.0000 | 10240.0 |
| VGGSS-144k | AUD20 | 0.6070 | 0.3930 | 0.6070 | 0.3930 | 0.5035 | 0.1011 | 1.0000 | 10240.0 |
| VGGSS-144k | IMG20 | 0.6095 | 0.3905 | 0.6095 | 0.3905 | 0.5067 | 0.1003 | 1.0000 | 10240.0 |
| VGGSS-144k | Agreement P | 0.6231 | 0.3769 | 0.6247 | 0.3753 | 0.4577 | 0.0822 | 1.0000 | 8825.7 |
| VGGSS-144k | AUD-extra candidate | 0.4895 | 0.5105 | 0.4960 | 0.5040 | 0.0458 | 0.0189 | 0.7854 | 1414.3 |
| VGGSS-144k | IMG-extra | 0.5129 | 0.4871 | 0.5147 | 0.4853 | 0.0490 | 0.0181 | 0.7854 | 1414.3 |
| Flickr-144k | Random A-matched | 0.5069 | 0.4931 | 0.5069 | 0.4931 | 0.1988 | 0.2064 | 1.0000 | 10240.0 |
| Flickr-144k | AUD20 | 0.8511 | 0.1489 | 0.8511 | 0.1489 | 0.3613 | 0.0595 | 1.0000 | 10240.0 |
| Flickr-144k | IMG20 | 0.8415 | 0.1585 | 0.8415 | 0.1585 | 0.3576 | 0.0650 | 1.0000 | 10240.0 |
| Flickr-144k | Agreement P | 0.8602 | 0.1398 | 0.8618 | 0.1382 | 0.2960 | 0.0441 | 1.0000 | 8175.6 |
| Flickr-144k | AUD-extra candidate | 0.8145 | 0.1855 | 0.8087 | 0.1913 | 0.0653 | 0.0154 | 0.9120 | 2064.4 |
| Flickr-144k | IMG-extra | 0.7733 | 0.2267 | 0.7612 | 0.2388 | 0.0617 | 0.0209 | 0.9120 | 2064.4 |

## Enrichment

| Dataset | Scope | FG(P) | FG(A20) | FG(I20) | P-A lift | P-I lift | BG(AUD-extra) | BG(A20) | BG(Random matched) | NA-A lift | NA-Random lift |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VGGSS-144k | Macro | 0.6231 | 0.6070 | 0.6095 | 0.0161 | 0.0135 | 0.5105 | 0.3930 | 0.6772 | 0.1174 | -0.1667 |
| VGGSS-144k | Micro | 0.6247 | 0.6070 | 0.6095 | 0.0178 | 0.0152 | 0.5040 | 0.3930 | 0.6641 | 0.1109 | -0.1601 |
| Flickr-144k | Macro | 0.8602 | 0.8511 | 0.8415 | 0.0091 | 0.0186 | 0.1855 | 0.1489 | 0.4822 | 0.0366 | -0.2967 |
| Flickr-144k | Micro | 0.8618 | 0.8511 | 0.8415 | 0.0107 | 0.0203 | 0.1913 | 0.1489 | 0.4773 | 0.0424 | -0.2860 |

## Purity-Coverage Tradeoff

| Dataset | Top-k | P FG purity | P FG recall | P empty | AUD-extra BG purity | AUD-extra BG coverage | AUD-extra empty |
|---|---:|---:|---:|---:|---:|---:|---:|
| VGGSS-144k | 10% | 0.6753 | 0.2429 | 0.0021 | 0.4166 | 0.0119 | 0.2726 |
| VGGSS-144k | 20% | 0.6231 | 0.4577 | 0.0000 | 0.5105 | 0.0189 | 0.2146 |
| VGGSS-144k | 30% | 0.5724 | 0.6138 | 0.0000 | 0.5982 | 0.0254 | 0.1712 |
| Flickr-144k | 10% | 0.8772 | 0.1374 | 0.0080 | 0.1613 | 0.0105 | 0.1800 |
| Flickr-144k | 20% | 0.8602 | 0.2960 | 0.0000 | 0.1855 | 0.0154 | 0.0880 |
| Flickr-144k | 30% | 0.8341 | 0.4575 | 0.0000 | 0.2475 | 0.0189 | 0.0920 |

## Confidence Quartiles

| Dataset | Score | Q1 | Q2 | Q3 | Q4 |
|---|---|---:|---:|---:|---:|
| VGGSS-144k | P FG purity | 0.5146 | 0.5994 | 0.6641 | 0.7208 |
| VGGSS-144k | AUD-extra BG purity | 0.4666 | 0.4682 | 0.5153 | 0.5659 |
| Flickr-144k | P FG purity | 0.8274 | 0.8487 | 0.8863 | 0.8847 |
| Flickr-144k | AUD-extra BG purity | 0.1978 | 0.1978 | 0.1628 | 0.2070 |

## Stage1 Hard Cases

| Dataset | Group | Count | P FG purity | P recall | P non-empty | AUD-extra BG purity | BG coverage | AUD-extra non-empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VGGSS-144k | ALL | 5158 | 0.6231 | 0.4577 | 1.0000 | 0.5105 | 0.0189 | 0.7854 |
| VGGSS-144k | IMG_ONLY | 168 | 0.7707 | 0.5418 | 1.0000 | 0.5752 | 0.0210 | 0.7440 |
| VGGSS-144k | AUD_ONLY | 271 | 0.8145 | 0.3309 | 1.0000 | 0.2315 | 0.0132 | 0.7491 |
| VGGSS-144k | BOTH_SUCCESS | 1931 | 0.9083 | 0.3781 | 1.0000 | 0.2066 | 0.0105 | 0.7840 |
| VGGSS-144k | BOTH_FAIL | 2788 | 0.3980 | 0.5202 | 1.0000 | 0.7407 | 0.0251 | 0.7923 |
| VGGSS-144k | OGL_RESCUE | 357 | 0.7148 | 0.5297 | 1.0000 | 0.5090 | 0.0182 | 0.7507 |
| VGGSS-144k | IMG_ONLY_SHRINK | 125 | 0.7583 | 0.5756 | 1.0000 | 0.5863 | 0.0207 | 0.7760 |
| Flickr-144k | ALL | 250 | 0.8602 | 0.2960 | 1.0000 | 0.1855 | 0.0154 | 0.9120 |
| Flickr-144k | IMG_ONLY | 13 | 0.7776 | 0.4273 | 1.0000 | 0.3985 | 0.0244 | 1.0000 |
| Flickr-144k | AUD_ONLY | 15 | 0.7666 | 0.2427 | 1.0000 | 0.2585 | 0.0148 | 0.8000 |
| Flickr-144k | BOTH_SUCCESS | 188 | 0.9218 | 0.2831 | 1.0000 | 0.1232 | 0.0118 | 0.9255 |
| Flickr-144k | BOTH_FAIL | 34 | 0.5924 | 0.3405 | 1.0000 | 0.4336 | 0.0320 | 0.8529 |
| Flickr-144k | OGL_RESCUE | 19 | 0.7642 | 0.3927 | 1.0000 | 0.3462 | 0.0255 | 1.0000 |
| Flickr-144k | IMG_ONLY_SHRINK | 10 | 0.8029 | 0.4689 | 1.0000 | 0.4618 | 0.0262 | 1.0000 |

IMG_ONLY+SHRINK Stage1 vs Stage2 (Stage2 is diagnostic only):

- VGGSS-144k: Stage1 P/NA purity `0.7583` / `0.5863`; Stage2 P/NA purity `0.8060` / `0.5331`; Stage1/Stage2 NA coverage `0.0207` / `0.0188`.
- Flickr-144k: Stage1 P/NA purity `0.8029` / `0.4618`; Stage2 P/NA purity `0.8611` / `0.2710`; Stage1/Stage2 NA coverage `0.0262` / `0.0131`.

## Correctness And Ranking Viability

- VGGSS-144k correctness fractions: `{'P->FG': 0.6247369897693787, 'P->BG': 0.3752630102306213, 'NA->FG': 0.4960124611787619, 'NA->BG': 0.503987538821238, 'NI->FG': 0.5146813368542953, 'NI->BG': 0.48531866314570465}`.
- VGGSS-144k `FG purity(P)-FG purity(AUD-extra)`: mean `0.1316`, median `0.0842`, std `0.3241`, fraction `>0` `0.5932` over `4051` valid samples.
- Flickr-144k correctness fractions: `{'P->FG': 0.8617924325212926, 'P->BG': 0.1382075674787074, 'NA->FG': 0.8086790054563492, 'NA->BG': 0.19132099454365079, 'NI->FG': 0.7612130301339286, 'NI->BG': 0.23878696986607142}`.
- Flickr-144k `FG purity(P)-FG purity(AUD-extra)`: mean `0.0514`, median `0.0000`, std `0.2640`, fraction `>0` `0.3816` over `228` valid samples.

## Stage1 Versus Stage2

| Dataset | Stage1 P FG | Stage2 P FG | Stage1 AUD-extra BG | Stage2 AUD-extra BG | P Pearson/Spearman | AUD-extra Pearson/Spearman |
|---|---:|---:|---:|---:|---:|---:|
| VGGSS-144k | 0.6231 | 0.6506 | 0.5105 | 0.4475 | 0.9838/0.9773 | 0.7696/0.7672 |
| Flickr-144k | 0.8602 | 0.8984 | 0.1855 | 0.1107 | 0.9273/0.8476 | 0.5980/0.5293 |

## Qualitative

- VGGSS-144k deterministic selections: `{'BOTH_FAIL': '-0BIyqJj9ZU_000030', 'HIGH_PURITY_P': '-2-wdcN5vOw_000017', 'LOW_PURITY_P': '-W3WpZvJX2o_000027', 'HIGH_PURITY_NA': '-3Kv4fdm7Uk_000030', 'NA_TRUE_EXTENT': '-2-wdcN5vOw_000017', 'BOTH_SUCCESS': '-0UuUoXQUoI_000107', 'OGL_RESCUE': '-4bPiXbovf0_000008', 'AUD_ONLY': '-GW1J75oAKU_000304', 'IMG_ONLY_SHRINK': '-hYRFCQdbLg_000030'}`.
- Flickr-144k deterministic selections: `{'BOTH_SUCCESS': '10000130166', 'HIGH_PURITY_P': '10000130166', 'LOW_PURITY_P': '5237296528', 'HIGH_PURITY_NA': '10278084464', 'NA_TRUE_EXTENT': '10000130166', 'BOTH_FAIL': '10008553263', 'AUD_ONLY': '10013411946', 'IMG_ONLY_SHRINK': '10548273474', 'OGL_RESCUE': '10548273474'}`.
- Agreement seeds generally retain the shared response core and remove weak branch-specific fringes. Their purity increases monotonically with agreement confidence on VGG; Flickr saturates at high confidence.
- AUD-extra is not a clean context mask. It is background-enriched inside IMG_ONLY/SHRINK and BOTH_FAIL subsets, but in AUD_ONLY and BOTH_SUCCESS it frequently contains real object extent.
- Flickr is the decisive failure mode for negative supervision: the global Stage1 AUD-extra candidate remains foreground-dominated, despite a small background-rate lift over AUD20.

## Decision

**Case B - Positive Seeds Work, Negative Seeds Fail.**

Agreement P is consistently better than both single-branch Top20 seeds, stays non-empty on all samples, and has non-trivial recall. This supports sparse positive consistency.

The context-negative candidate does not meet the purity requirement: although `BG(AUD-extra)-BG(AUD20)` is positive on both datasets, matched-random seeds have substantially higher background purity, Flickr AUD-extra is only about 18.5% background, and ranking viability is below 50% on Flickr. AUD_ONLY contamination is also severe.

**Agreement positive supervision supported; context-negative supervision unsupported. Next: positive-only supervision, not context suppression. Do not start 5.1 automatically.**
