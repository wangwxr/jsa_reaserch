# Experiment 2.4 - Object-Aware Multi-Geometry Spatial Specialization

This experiment has exactly two training stages:

```text
formal L3+L4 Stage1 checkpoint -> freeze
fresh G spatial student -> train Stage2 with four fixed losses
```

It never initializes from a trained 1.3G student. The trainable architecture is exactly the
formal G student: copied `proj3_spatial` plus the zero-final-conv `TopDownL3Adapter`, totaling
1,443,072 parameters. The frozen L4 final Slot Attention query reads both the Stage1 7x7 keys
and the learned G 14x14 keys; no q/k/v, Slot Attention, gate, or spatial-slot module is added.

The fixed Stage2 objective is:

```text
L = L_audio_coarse + L_audio_equiv + L_own_coarse + L_own_equiv
```

`L_audio_coarse` and `L_audio_equiv` are the unchanged formal G implementation.
`L_own_coarse` is teacher-to-student categorical KL between `OWN7` and
`avg_pool2d(OWN14, 2, 2)`. `L_own_equiv` is valid-mask symmetric KL between View A ownership
and exactly warped View B ownership. All four weights are fixed to 1.0.

Run the required smoke audit on both GPUs:

```bash
bash chuagnxindian/2.4_object_aware_multigeom_spatial_specialization/smoke_144k_two_gpus.sh
```

Run both formal 144k, 50-epoch jobs:

```bash
bash chuagnxindian/2.4_object_aware_multigeom_spatial_specialization/run_144k_two_gpus.sh
```

The primary checkpoint is selected by strict improvement in `AUD_OBJ cIoU`. A separate
diagnostic checkpoint is selected by strict improvement in `AUD_FINE cIoU`; cIoU ties are
logged with AUC but never replace the incumbent.
