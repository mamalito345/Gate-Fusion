# Control A: Stage 2 without a gate

**Question.** The Stage 1 -> Stage 2 shift (weaker branch up, stronger branch down) could come from the gate's cross term, or simply from 25 more epochs of head training plus choosing the best epoch on the validation set. This control removes only the cross term.

**What is held fixed.** Start from the same Stage 1 backbones and heads, freeze the backbones, same learning rates, warm-up, epochs, batch size, same "best epoch by mean validation accuracy" rule. **The one change:** both gate matrices are set to zero and frozen, so the forward pass is `feat_own + 0`.

**Run.** In the same Colab session as `notebooks/CUHKX_gate_experiment_EN.ipynb`, after Cell 16:

1. `cell_20_nogate_control_train.py`: trains the control for the six backbones and writes `*_stage2_nogate_predictions.npz`. It asserts that the gate stayed exactly at zero; if it did not, the run is invalid.
2. `cell_21_nogate_control_analyze.py`: prints, per backbone and for the weaker, stronger and averaged output, `S2 - S1`, `S2nogate - S1`, and `S2 - S2nogate` with a paired bootstrap 95% interval.

**How to read it.**
- `S2 - S2nogate` near 0 for all three outputs: the shift is not the gate's cross term.
- `S2nogate - S1` already positive for the weaker branch: most of the shift happens without a gate.
- An interval that excludes 0 on the weaker or stronger branch, in several backbones: the cross term does something the control does not.

**What it cannot show.** One training run per configuration. The interval covers sampling of the 628 validation clips, not training randomness. The original Stage 2 runs were unseeded and this control uses seed 0, so run-to-run noise is mixed into the difference. A seed-matched repeat (gate on and off, same seeds) is the fix and is planned.

**Status.** Code written. No results are recorded in this repository yet. Results will be added to `results/` whatever they are.
