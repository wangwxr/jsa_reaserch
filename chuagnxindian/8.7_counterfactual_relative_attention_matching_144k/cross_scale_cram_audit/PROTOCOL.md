# Frozen protocol

- Seed: `12345`; existing selected Stage-1 and best Stage-2 checkpoints only.
- Test split/order: exact existing VGG-SS (5,158) and Flickr (250) test order.
- Counterfactual audios: fixed Experiment-8.1 `set1_indices`, first eight for
  map swaps and first four for the distance audit.
- Object regions: fixed Experiment-8.0 GT/NearFP region indices.
- Threshold and readout definitions: unchanged 1.3G `.6` threshold and
  `AUD`, `IMG_QUERY`, `IQR`, `OBJ_PRIOR`, `OGL`, `EXTRA_IQR_OGL` definitions.
- Group A: fixed old residual-audit label `A_ogl_help` (`OGL IoU - AUD IoU >
  .10` for the 144k Mean-CRAM model); it is never recomputed.
- Bootstrap: 2,000 deterministic resamples for paired/sample summaries.

The newly added map extraction performs `torch.inference_mode()` only. The
test hardness audit uses no GT/OGL metadata while calculating distances;
labels are joined only afterward for analysis.
