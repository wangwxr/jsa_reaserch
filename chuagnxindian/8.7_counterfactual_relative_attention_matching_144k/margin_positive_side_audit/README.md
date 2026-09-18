# 8.7 Margin / Positive-Side Audit

Final diagnosis: **A-NEGATIVE, then P1 with an AUD-path P3 signature.**

Mean-CRAM's relative counterfactual constraint is generally satisfied on the
residual VGG examples and its margin does not predict OGL gain.  The remaining
failure is instead an overly broad *matched-positive AUD map*: target, NearFP,
and background are jointly activated.  OGL gains primarily by removing this
already-positive non-target activation.

The full report is `summary/margin_positive_summary.md`; all sample-level
data and reproducible figures are retained in the named subdirectories.
