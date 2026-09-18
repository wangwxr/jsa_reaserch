# Experiment 8.7 Implementation

The verified production path is unchanged from 8.6:

- `MUFASAL3L4` in `mufasa_ablation2_l3_l4_ablation/model_l3_l4.py`;
- original loss in `1mufasaslot/model_mufasa_jsa.py`;
- L4 A2V `ca_av` and V2V `ia_vv` in `l3_l4_slot_attention.py`.

The exact original losses and weights are InfoNCE `1.0`, reconstruction `0.1`,
diversity `0.1`, and full A2V+V2A matching `100.0`. `objective.py` loads that
8.6-equivalent decomposition and adds `softplus(D_pos - mean(D_neg_k))` only.
The positive V2V target is `detach()`ed. For each wrong audio, only the existing
audio encoder and audio Slot Attention branch are evaluated against the same
matched-image L4 visual keys; this creates no new trainable parameters.

Wrong-audio passes use the existing audio encoder in evaluation-BN mode, so CRAM
does not change the original Stage-1 running statistics merely by evaluating four
additional audios. Their gradients still reach the audio encoder and the shared
L4 visual keys. The four branches are activation-checkpointed one at a time to
fit the fixed batch size and `K=4` without changing the loss or sampler.

The C0 checkpoint, initialization, fixed epoch orders, and verified cached
evaluation audios are reused read-only from 8.6. New C1/C2/C3 checkpoints and all
new outputs live exclusively in this directory.
