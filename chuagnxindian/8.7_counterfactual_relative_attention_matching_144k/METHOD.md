# CRAM-v0

The 8.6 result says complete `L_match` is useful for spatial organization but
does not by itself prove object grounding. CRAM retains that positive anchor and
adds only a relative same-image test: the matched audio must reproduce the
detached visual intra-modal template better than four wrong audios.

It does not prescribe where a wrong-audio heatmap should be, force it to zero,
or use pseudo masks. A lower wrong-audio similarity is treated as mechanism
evidence only when semantic GT/NearFP and natural localization tests also pass.
