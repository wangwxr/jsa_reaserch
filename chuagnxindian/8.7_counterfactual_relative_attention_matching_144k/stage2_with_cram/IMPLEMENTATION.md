# Implementation notes

`model.py` subclasses the original
`1.3G-multigeom_equivariant_l3_refine/MultiGeometryEquivariantRefinement`.
It exposes only the existing frozen C3 visual query and the existing trainable
K34 key tensor needed to calculate the exact relative CRAM term at 14x14.

The original `spatial_losses()` function remains the source of `L_coarse` and
`L_equiv`; `stage2_cram_components()` is additive and has no effect unless its
returned loss is explicitly weighted by `train.py`.

Checkpoints preserve the normal Stage-2 student state names
`proj3_spatial_state_dict` and `topdown_adapter_state_dict`, plus CRAM metadata.
They can therefore be evaluated by a dedicated loader without changing the
existing successful C3 Stage-2 checkpoint or its evaluator.

