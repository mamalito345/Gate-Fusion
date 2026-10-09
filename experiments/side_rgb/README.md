# Side hypotheses: RGB and sensor structure

Not part of the main result. See [docs/side-hypotheses.md](../../docs/side-hypotheses.md) for the hypothesis, what the code already implies, and the status of each item.

Planned (none run in this repository yet):

1. Real-colour Depth on one backbone (mvit_v2_s), Stage 1 Depth only, against the gray-replicated baseline.
2. First-convolution channel weights across the six backbones. A weight statistic, not proof of use. Inputs have three identical channels, so any preference should come from pretraining.
3. Per-clip colour-drift score against per-clip Depth-branch errors.
4. Minimal initialisation ablation (pretrained RGB start against random init) for IR and Depth.
