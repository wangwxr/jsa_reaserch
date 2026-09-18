# 8.7 Margin / Positive-Side Audit — Final

## Decision

**Phase A: A-NEGATIVE.  Phase B: Case P1 — matched-positive map over-activation;
with a secondary P3-style indication that the AUD readout is broader than
IMG_QUERY.**

The remaining VGG OGL gap is not explained by failure to satisfy Mean-CRAM's
relative positive-vs-negative constraint.  It is explained by the positive
AUD map activating target and non-target context together.  This is still the
same object-level discrimination problem, but the remaining lever is the
**positive spatial side**, not negative aggregation, mining, or
sample-selective violation weighting.

## Phase A — margin does not identify Group-A

| metric | Group-A (n=428) | non-Group-A (n=4,730) | A − non-A 95% CI | MW p |
|---|---:|---:|---:|---:|
| margin mean | 3.14e-5 | 3.18e-5 | [-0.80e-5, 0.60e-5] | .626 |
| margin median | 1.78e-5 | 1.77e-5 | — | — |
| violation rate | .1869 | .2047 | [-.0545, .0219] | .383 |
| softplus loss | .693131 | .693131 | [-3e-6, 4e-6] | .626 |
| normalized margin | .2704 | .2598 | [-.0206, .0398] | .682 |

The directions are incompatible with a violation explanation: Group-A has a
slightly *lower*, not higher, violation rate.  Margin versus OGL-AUD IoU is
rho=.022 (95% bootstrap CI [-.006,.047], p=.114); violation versus OGL gain is
rho=.001.  There is no monotonic bucket pattern: Group-A fractions span
.075–.101 across margin quintiles and OGL gain does not rise toward low margin.

The controls show raw distance scale dependence, as expected (VGG violation
.243 at 10k and .203 at 144k; Flickr .516 at 10k and .276 at 144k).  They do
not change the within-model VGG conclusion.  Therefore **negative-side
violation weighting is stopped**.

## Phase B — the matched AUD map is broad, not incomplete

Fixed Group-A versus non-Group-A, on matched AUD maps:

| metric | Group-A | non-Group-A | A − non-A, 95% CI |
|---|---:|---:|---:|
| GT response | .7974 | .7599 | [.0282,.0480] |
| NearFP response | .7339 | .7191 | [.0064,.0228] |
| background response | .4914 | .4527 | [.0306,.0475] |
| precision | .4652 | .5441 | [-.0978,-.0591] |
| coverage | .8608 | .8064 | [.0333,.0746] |
| activated area | .4552 | .4280 | [.0147,.0391] |
| false-positive area | .2477 | .1965 | [.0384,.0630] |
| entropy | .9914 | .9902 | [.0009,.0016] |

Thus the residual is not missing target extent: coverage is *higher*.  It is
an over-broad positive decision with lower precision and more outside-GT
activation.  GT-NearFP gap is also higher (.0649 vs .0437), so the failure is
not simply a globally collapsed GT/NearFP score gap; it is that both regions
receive excessive positive response, together with broad background response.

## The decisive satisfied-but-over partition

81.3% of fixed Group-A is CRAM-satisfied.  Of all Group-A examples, **238/428
(55.6%) are both CRAM-satisfied and frozen-OVER**; 238/287 (82.9%) of
Group-A's OVER cases are CRAM-satisfied.  The Group-A fraction is .1107 in the
satisfied+OVER partition, nearly twice .0561 in satisfied+not-OVER.  This is
the direct evidence that satisfying the relative cross-audio constraint does
not decompose the positive map into target versus context.

## What OGL changes on these same positive maps

For Group-A, OGL changes GT response by only -0.0093 but NearFP response by
**-0.0597**, and reduces activated area by **-0.1077**.  Its paired pixel
transition is dominated by **RemoveFP=6,185** versus RemoveTP=355 (mean);
AddTP is not the central source of its gain.  Therefore OGL is mainly pruning
context/false-positive response already present under matched audio, not
repairing an audio-swap/counterfactual mismatch.

## AUD / IMG_QUERY / IQR on fixed Group-A

The native-grid descriptive replay gives:

| readout | precision | coverage | activated area | GT-NearFP gap |
|---|---:|---:|---:|---:|
| AUD | .470 | .854 | .446 | .068 |
| IMG_QUERY | .492 | .801 | .397 | .058 |
| IQR | .482 | .836 | .423 | .064 |

IMG_QUERY is more compact and more precise than AUD; IQR lies between them.
This is a **P3-style AUD-path signature**: matched audio adds useful coverage,
but also the excessive spatial spread.  It does not overturn P1, which is the
primary mechanism classification.

## Visual ambiguity control

Frozen C3 GT–NearFP visual-key cosine is .9139 for Group-A and .8970 for
non-Group-A.  The mean difference CI is [.0069,.0253], but the distributional
Mann–Whitney test is non-significant (p=.345; rank-biserial=.028).  This is not
robust evidence for P2/shared visual representation confusion.  Do not base a
new method on it.

## Final methodological decision

1. **Mean-CRAM remains the main paper method.**
2. Hard aggregation, tau sweeps, large-pool mining, hardness gating, and now
   sample-wise violation weighting can be formally stopped.
3. The VGG residual remains within the same target-vs-context discrimination
   problem; it is not evidence that a second innovation is necessary.
4. The evidence supports: CRAM fixes useful cross-audio relative structure,
   but does not yet completely regulate **matched-positive target/context
   separation**.
5. The next CRAM version should address the **positive spatial side**.  It
   needs a label-free spatially selective positive anchor/weighting that
   preserves the high Group-A coverage while reducing context activation.
6. No method is implemented or trained by this audit.

## One main direction and one backup (not launched)

**Main:** positive-side spatially selective CRAM.  Its eventual loss should
retain Mean-CRAM's same K=4 mean-negative term and Stage-1-only placement, but
weight the *positive anchor's spatial locations* rather than mine/aggregate
negatives.  The required pre-training design constraint is to demonstrate a
label-free positive-map confidence signal that is associated with broad
matched activation.  Freeze lambda=1, K=4, mean aggregation, data order,
architecture, seed, and vanilla Stage-2.  Success requires VGG precision and
RemoveFP to improve without material coverage/RemoveTP loss, while Flickr
remains AUD > OGL.

**Backup:** if no label-free positive-side signal can be validated, keep
Mean-CRAM frozen and stop CRAM-internal modifications.  Do not add a second
branch or innovation without a new diagnosis.
