# Causal object-support audit

Main unit: a 3x3 patch on frozen F34 (14x14). Candidate perturbations are token-zero and per-image local-mean replacement. A preliminary random-patch sanity selects one by lower global compatibility displacement and lower feature-norm artifact; that choice is frozen before Group-A labels are read. No optimization, parameter update, GT/OGL-guided intervention, or external model is used.
