[← README](../README.md)

# The mathematics: what we did

Everything below is what the code does, nothing more. Symbols follow the notebook.

## Setup

A clip is a pair x = (x_IR, x_D) of 16 frames each, with a label y in {1, …, 40}. A backbone φ maps a clip to a feature vector in R^d. Two copies of the same architecture are used, φ_IR and φ_D. Let f = φ_IR(x_IR) and g = φ_D(x_D).

## Stage 1: independent models

```text
z_IR = W_IR f + b_IR
z_D  = W_D  g + b_D
L    = CE(z_IR, y) + CE(z_D, y)
```

The two models share no parameters, so this is the same as training each alone.

## The plain average

```text
p = ( softmax(z_IR) + softmax(z_D) ) / 2
prediction = argmax p
```

This is the baseline. "Fusion" in every table means this operation, at every stage.

## The gate

```text
f~ = f + A g          A in R^(d x d), no bias, A = 0 at start
g~ = g + B f          B in R^(d x d), no bias, B = 0 at start
z~_IR = W_IR f~ + b_IR
z~_D  = W_D  g~ + b_D
```

At A = B = 0 the model is exactly the two Stage 1 models, because the heads are initialised from them. There is no single fused output: the gate produces two sets of logits, one per branch. They are averaged exactly as above.

## Stages 2 and 3

```text
L = CE(z~_IR, y) + CE(z~_D, y)

Stage 2: trainable = {A, B, W_IR, b_IR, W_D, b_D},  backbones frozen
Stage 3: trainable = everything
```

Two AdamW optimisers step disjoint parameter groups (IR side and Depth side) with the same schedule. Because AdamW acts on each parameter separately, this equals one AdamW on L. The IR backbone receives gradient from both losses in Stage 3, the second one through B f. The gate A appears only in z~_IR, and B only in z~_D.

## What we measure

```text
acc(p)      = (1/N) sum_i 1[ argmax p_i = y_i ],   N = 628
oracle      = (1/N) sum_i 1[ argmax softmax(z_IR,i) = y_i  or  argmax softmax(z_D,i) = y_i ]
Delta_S2    = acc(p at Stage 2) - acc(p at Stage 1)
Delta_S3    = acc(p at Stage 3) - acc(p at Stage 1)
```

Per-backbone Δ is a difference of two accuracies on the same 628 clips, so it is a paired difference. The standard error of a single accuracy is sqrt(a(1−a)/N), about 2.0 points near 60%. The reported summaries are the mean, the median and the count of positive Δ over six backbones.

## The controls

**Control A.** Run Stage 2 again from the same Stage 1 models with A = B = 0 frozen. Then z~ = z, and only the heads train. Same learning rates, same 25 epochs, same epoch-selection rule. The difference between Stage 2 and this run is what the cross terms add, holding head retraining and epoch selection fixed. Code: [experiments/01_nogate_control](../experiments/01_nogate_control/).

Paired comparison on the same clips:

```text
d_i   = 1[ correct under run 1 ] - 1[ correct under run 2 ]
delta = mean(d_i) * 100
95% interval: resample the 628 clips with replacement 4000 times,
              take the 2.5th and 97.5th percentiles of mean(d) * 100
```

This interval covers the sampling of validation clips only. It does not cover training randomness. Control A is one run per configuration, and the original Stage 2 was unseeded while the control uses seed 0, so run-to-run noise is mixed into the difference. Seed-matched repeats are the fix and are listed as planned.

**Control B (not yet written).** Stage 3 with A = B = 0 frozen: 25 more epochs of independent end-to-end training. If it recovers most of the Stage 3 gain, the gain was training length, not the gate.


## Analysis: what the gate can express

Written out in full, with proofs and a numerical check, in [cross-gate-hypothesis.pdf](cross-gate-hypothesis.pdf). The results, in brief:

1. **Logit identity.** z~_IR = W_IR f + (W_IR A) g + b_IR. Each branch is an affine classifier on the concatenation [f; g]; the weight on g is the product V = W_IR A.
2. **Capacity with frozen backbones (Stage 2).** If W_IR has full row rank (d ≥ 40 here, generically true), every V is reachable: A = W_IR^T (W_IR W_IR^T)^(-1) V. So Stage 2 can represent exactly what two concatenation heads can, and nothing more. Any Stage 2 difference to a concatenation head must come from initialisation, optimisation geometry or implicit regularisation. That head is the missing early-fusion control (Control C).
3. **Additive, not interactive.** Logits are a sum of a term in f and a term in g. The gate cannot express "trust IR only when depth shows X". Interaction can only appear through the learned features, i.e. in Stage 3.
4. **Gradients.** dL/dA = W^T (p − e_y) g^T, dL/dB likewise; dL/df = W_IR^T r_IR + B^T W_D^T r_D, dL/dg = A^T W_IR^T r_IR + W_D^T r_D. At A = B = 0 the gate gradient is non-zero (zero initialisation is not a saddle). The extra terms in dL/df and dL/dg are the only coupling between the branches, and they are present only when the backbones are trained (Stage 3).
5. **Preconditioning.** Under plain gradient flow on A, V = W A moves as dV/dt = −W W^T (p − e_y) g^T, i.e. concatenation-weight training preconditioned by W W^T. AdamW is not covered by this statement.
6. **Averaging and diversity.** Because both branches see [f; g] after Stage 2, branch accuracies should converge and errors become more alike. The data agree: mean branch gap 5.4 → 1.9 (S2) → 3.1 (S3) points; mean oracle 65.2 → 63.9 → 64.3. So the earlier "weaker branch rises, stronger falls" pattern is what convergence predicts, with or without any transfer of information.
7. **Statistics.** Stage 3: mean +2.18, SE 1.24, t = 1.75 on 5 df (p ≈ 0.14, optimistic because the six backbones share clips, pretraining and the selection rule); sign test 4/6, p ≈ 0.69. Stage 2: t = 0.32.

`verify_math.py` checks items 1, 2 and 4 against finite differences (largest discrepancy 8e-10).

Control C (not yet written): on frozen Stage 1 features train, per branch, one linear head on [f; g] with the same optimiser, schedule and epoch selection. Control D: IR with IR and depth with depth, same architecture.
