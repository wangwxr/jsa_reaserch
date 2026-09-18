# Frozen compatibility score

The main scalar is the negative of the existing original L_match A2V positive distance:

`s(I,A) = -mean_p ( A2V(I,A)[p] - stopgrad(V2V(I)[p]) )^2`.

It is not a new learned score. It is exactly the existing matched audio-to-visual spatial matching semantic (with sign reversed so higher is better). Intervention is performed at the frozen Stage-2 F34 14x14 visual feature, followed by the existing fine-key and A2V computation. The V2V anchor stays fixed from the unperturbed image, making the drop comparable across local interventions.
