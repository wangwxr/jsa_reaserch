# Cross-scale CRAM audit: conclusion

## Main result

Mean-CRAM remains the strongest and most stable completed 8.7 variant.

| Scale / dataset | Mean AUD | Hard-default AUD | Hard-adjusted AUD | Mean AUD − OGL |
|---|---:|---:|---:|---:|
| 10k VGG-SS | .4259 / .4209 | .4162 / .4182 | — | -.0337 / -.0165 |
| 10k Flickr | .7840 / .6190 | .7960 / .6252 | — | -.0720 / -.0248 |
| 144k VGG-SS | .4343 / .4296 | .4275 / .4283 | .4254 / .4280 | -.0180 / -.0104 |
| 144k Flickr | .8960 / .6532 | .8720 / .6430 | .8480 / .6558 | +.0080 / +.0102 |

The complete six-readout table is `performance/all_readouts.csv`.

## What global soft-min does

- **10k VGG-SS:** Hard-default − Mean is −.0097 cIoU / −.0027 AUC. It raises
  coverage (+.0221) and activated area (+.0345), but lowers precision
  (−.0164) and GT–NearFP gap (−.0054): expansion, not selective suppression.
- **10k Flickr:** it has a real but modest local/global benefit over Mean
  (+.0120 cIoU / +.0062 AUC), while lowering precision and increasing area.
  It remains below the 10k 1.3G AUD cIoU (.804).
- **144k VGG-SS:** it improves precision (+.0047) and mildly shrinks area,
  with an essentially neutral gap/AUROC change, but loses coverage (−.0080)
  and AUD cIoU (−.0068).
- **144k Flickr:** it lowers gap (−.0143), AUROC (−.0152), precision
  (−.0109), and cIoU (−.0240). This is not the same useful VGG Group-A effect.

Thus 10k and 144k do **not** have one uniform Hard failure mode. Hardening is
scale- and dataset-sensitive, which is exactly why it cannot replace Mean as
the main formulation.

## Fixed VGG residual Group A

For the frozen 428 OGL-help samples, Mean has mean IoU .4106. Default Hard
raises it to .4292, raises precision .4652→.4893 and gap .0649→.0684, with
coverage .8608→.8559. Adjusted VGG tau is weaker on this subgroup: IoU .4244,
precision .4822, gap .0628. Yet this local gain cannot compensate for global
damage: default Hard has 114 `Hard-help` samples but 141 `Hard-hurt` samples.
The hurt samples began much stronger (Mean IoU .5885, precision .7725) than
the help samples (Mean IoU .4562, precision .6235). This is unwanted spatial
redistribution/over-correction, not a simple handful of catastrophic cases.

## Tau verdict

The adjusted tau changed the intended concentration (VGG stronger, roughly
N_eff 2; Flickr smoother, roughly N_eff 3.08), but not the failure class.
VGG adjusted is below default globally and in Group A. Flickr adjusted raises
AUC over default (.6558 vs .6430) but loses cIoU (.848 vs .872), further
lowers GT–NearFP separation, and keeps AUD below OGL. Therefore: **B — tau
changes degree, not the failure nature. Stop global soft-min tau sweeps.**

## Internal hardness signal

At VGG-SS 144k, Group A vs all other samples has no reliable separation on
the proposed GT-free signals: `mean_minus_min` CI crosses zero
(+3.62e-6, [−1.80e-6, +1.00e-5], Mann–Whitney p=.077), and N_eff/entropy also
cross zero. Group A has a slightly lower violation rate for
`D_pos >= D_mean` (.187 vs .205), not a higher one.

There is a weaker *contrast* between OGL-help and AUD-better subsets (e.g.
mean-minus-min p=.010 and N_eff p=.014), but it does not generalize from
Group A to non-Group-A. It is insufficient as an Adaptive-CRAM gate.

Mean itself is not saturated: at test time about 20.3% of VGG-144k samples
still violate `D_pos < D_mean`; 39.4% have a negative closer than the positive
while their **mean** negative remains farther. However, these violations are
not enriched in the residual Group A, so violation-only weighting is not a
justified primary route to close the VGG OGL gap.

## Decision

1. **Keep Mean-CRAM as the main method.** It is the only completed stable
   variant that yields Flickr AUD > OGL at 144k and retains the established
   two-stage object-discrimination gain.
2. **Keep Hard-default as a negative ablation.** Its Group-A gain is useful
   mechanistic evidence, but not a main-method candidate.
3. **Do not put adjusted tau in the main text.** At most report it in the
   supplement as the preregistered one-step tau check demonstrating that tau
   scaling does not resolve the global trade-off.
4. The paper is close but not closed: VGG Mean AUD remains .0180 cIoU below
   its OGL readout. No evidence from this audit requires a second module.
