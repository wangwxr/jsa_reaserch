# One permitted post-default temperature adjustment

Only `tau` changes.  The model, seed (12345), K=4 frozen wrong-audio mapping,
lambda=1, dataset, order, initialization, all original losses and vanilla
Stage-2 recipe are unchanged.  Both variants use the independently verified
exact audio-token reuse implementation; full B=256 checks gave zero loss and
gradient difference and about 2.25x CRAM-step speedup.

| Dataset | Variant | tau | Expected N_eff | Reason |
|---|---|---:|---:|---|
| VGG-SS | `vggss_tau_stronger_neff2` | 1.3987111388194196e-5 | 2.00 | Group-A precision/gap improved with coverage retained, but residual context suppression remains insufficient. |
| Flickr | `flickr_tau_weaker_neff3_08` | 1.7400470535358854e-5 | 3.08 | Default soft-min reduced gap, AUROC and precision; soften it toward the validated Mean limit. |

Each variant is isolated and automatically runs `Stage-1 -> vanilla Stage-2
-> B=256 acceleration validation`.  No further tau, lambda, K or warmup sweep
is authorized by this protocol.
