# Experiment 8.7 Decision

## CRAM-NO-GO

Original mechanism gate: `stage2_allowed = false`.

2026-09-12 amendment: the user explicitly authorized C3 Stage 2 on both
datasets as a performance follow-up. Current execution permission:
`stage2_allowed = true` (user override, not a passed CRAM-GO gate).
See `stage2/PROTOCOL.md`. The Stage-1 mechanism decision remains unchanged.

CRAM-v0 preserves much of the positive spatial anchor and avoids the 8.6
No-L_match collapse. It does **not** consistently lower matched/wrong map
Spearman, improve AUD-matched GT-vs-NearFP S_CF, and preserve matched-coverage
precision across VGG-SS and Flickr at the same time.

Flickr C1/C3 raw NearFP separation is a useful dataset-specific hint, but is
not replicated on VGG-SS and does not establish new conditional grounding.
This is neither CRAM-GO nor CRAM-WEAK-GO under the preregistered gate.

Stage 2 follows the unchanged 1.3G recipe under the explicit amendment above.
No second seed or replacement loss is started automatically.
