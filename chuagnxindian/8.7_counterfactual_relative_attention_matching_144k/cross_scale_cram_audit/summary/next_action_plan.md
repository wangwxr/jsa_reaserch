# One data-supported next action

## Main: improve negative construction while retaining Mean aggregation

Keep the exact Mean-CRAM loss and Stage-2 fixed:

\[
L_{CRAM}=\operatorname{softplus}(D_{pos}-\operatorname{mean}_k D_{neg,k}).
\]

Replace only the source of one or a small fixed portion of the K=4 negatives:
select an **audio-representation-near, different-video** counterfactual from
the training pool (using frozen/current audio-query similarity without labels),
and retain the other negatives as the existing uniform fixed draws. This is
not global soft-min and adds no architecture or stage.

Why this is the main direction:

- Default Hard contains valuable Group-A suppression signal, so the residual
  is not inconsistent with the CRAM family.
- It hurts samples that Mean already localizes well, proving that an
  *every-sample* hardest-distance pressure is the wrong selector.
- D-distance hardness cannot identify Group A versus all other samples; a
  distance-only adaptive gate is unsupported.
- The remaining plausible CRAM-internal lever is therefore **negative
  relevance**, not a third tau/lambda/K sweep.

Minimal next experiment: exactly one Mean-aggregation variant with K=4,
lambda=1, same initialization/order/seed/architecture/Stage-2; change only
one negative slot to a deterministic different-video nearest audio-query
negative. First audit false-negative rate and distance diversity without test
metrics. Success requires VGG AUD cIoU >= .4523 (the Mean OGL cIoU), no loss
of Flickr AUD > OGL, VGG Group-A precision/gap improvement, and no global
coverage collapse.

## Backup: smooth Mean violation weighting, not negative soft-min

Only if the main negative-relevance diagnostic shows no extra diversity,
consider a single sample-wise weighting of the existing Mean loss:

\[
w_i=\operatorname{stopgrad}\{\sigma[(D_{pos,i}-D_{neg,mean,i})/T]\},
\quad L=\operatorname{mean}_i w_i\,\operatorname{softplus}(D_{pos,i}-D_{neg,mean,i}).
\]

Freeze K, lambda, all Stage-2 code, and the uniform negative construction.
This only de-emphasizes already-satisfied pairs. It is a backup because the
current violation signal is not enriched in Group A; it should not be sold as
an OGL-gap-targeted intervention without a new diagnostic.
