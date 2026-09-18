# Protocol

Mean-CRAM is the frozen reference: `D_neg = mean_k D_neg,k`, K=4, lambda=1.0.
Hard-CRAM will change only this aggregation to a smooth soft-min:

`D_neg_hard = -tau * log(mean_k exp(-D_neg,k/tau))`.

All positive terms, detached visual target, wrong-audio mapping, architecture,
seed, initialization, data order, original losses and vanilla Stage-2 recipe
remain unchanged. Tau is selected from the recorded C3 distance distribution;
the desired default effective-negative number is approximately 2.5.
