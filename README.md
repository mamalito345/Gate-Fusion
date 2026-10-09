# Beyond Early and Late Fusion: A 3-Stage Cross-Gate Architecture for Small-Model HAR

Companion repository for the poster at the **CUHK-X Challenge Grand Finals** (Small Model Track, Shanghai, 12 October 2026).
Melih Talha Bektaş, Üsküdar University.

Two video models watch the same person, one through an infrared (IR) camera, one through a depth camera. The usual way to combine them is to average their outputs (late fusion). Here the two models first pass information to each other through a small learned cross-gate, and we ask whether that beats the plain average. A formal analysis of what the architecture can and cannot express, and of what the evidence supports, is in **[docs/cross-gate-hypothesis.pdf](docs/cross-gate-hypothesis.pdf)**.

> **Short answer so far.** Trained end to end, gate then average beat the plain average of two independent models by **+2.2 points on average, on 4 of 6 backbones**. Training only the gate and the classifier heads did not: +0.5 on average, −0.5 median, 2 of 6. One accuracy on 628 validation clips has a standard error of about 2 points.
> Whether the gate is the cause of the end-to-end gain is what the control experiments decide. They are not finished.

**On this page:** [The system](#the-system) · [What the numbers say](#what-the-numbers-say) · [What moves inside](#what-moves-inside) · [What this does not show](#what-this-does-not-show) · [Where things stand](#where-things-stand) · [Read further](#read-further) · [Repository map](#repository-map) · [Reproducing](#reproducing)

## The system

```
              f = IR backbone(IR clip)          g = Depth backbone(Depth clip)

IR branch:     f + A g   ->  IR head     ->  softmax  --+
                                                        +-->  average  -->  prediction
Depth branch:  g + B f   ->  Depth head  ->  softmax  --+

A, B: one bias-free linear layer each, both zero at the start.
```

- Each backbone is a video network pretrained on Kinetics-400. Six were tried: `r2plus1d_18`, `r3d_18`, `mc3_18`, `s3d`, `mvit_v2_s`, `swin3d_t`.
- At the start A = B = 0, so the system is exactly two independent models.
- There is no single fused output. The gate gives two sets of logits, one per branch, and the two softmax outputs are averaged.

Training has three stages of 25 epochs each:

| stage | what is trained |
|---|---|
| 1 | two independent models, one per sensor. This is the plain-average baseline |
| 2 | start from Stage 1, freeze both backbones, train the gates and the two heads |
| 3 | unfreeze everything and train end to end |

The "fusion" number at every stage is the same operation, an average of two softmax outputs. Only what the two branches were trained to be differs between stages.

Data: [CUHK-X Small Model Track](https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Small_Model_Track), IR and Depth only, 40 action classes. Split by person: 2,303 training clips, 628 validation clips from four held-out subjects. Every number here is on those 628 clips.

## What the numbers say

Accuracy of the averaged output, in %, on 628 validation clips. Δ is the change from the plain average, in points.

| backbone | plain average (Stage 1) | gate only (Stage 2) | Δ | end to end (Stage 3) | Δ | best single model (Stage 1) |
|---|--:|--:|--:|--:|--:|--:|
| r2plus1d_18 | 59.2 | 59.6 | +0.3 | 58.0 | −1.3 | 60.5 |
| r3d_18 | 59.1 | 58.0 | −1.1 | 59.1 | 0.0 | 57.6 |
| mc3_18 | 48.1 | 47.1 | −1.0 | 51.8 | +3.7 | 50.5 |
| s3d | 45.4 | 52.5 | +7.2 | 52.4 | +7.0 | 45.5 |
| mvit_v2_s | 72.9 | 72.9 | 0.0 | 76.1 | +3.2 | 74.4 |
| swin3d_t | 60.8 | 58.1 | −2.7 | 61.3 | +0.5 | 59.7 |
| **mean** | | | **+0.5** | | **+2.2** | |
| **median** | | | −0.5 | | +1.8 | |
| positive on | | | 2 of 6 | | 4 of 6 | |

- **Gate only (Stage 2):** mean +0.5, median −0.5, positive on 2 of 6. No reliable gain.
- **End to end (Stage 3):** mean +2.2, median +1.8, positive on 4 of 6.
- **One backbone carries much of it.** `s3d` gains about 7 points at both stages and had the weakest baseline. Without it the means are −0.9 (gate only) and +1.2 (end to end).
- **Against the best single model of Stage 1** (not the average): gate only 0.0 on average; end to end +1.7 on average, positive on 5 of 6.
- **Noise.** One accuracy on 628 clips has a standard error of about 2 points. A paired difference on one backbone is less noisy than that, but still hard to read alone. What carries information is the pattern across six backbones.

## What moves inside

Accuracy of each branch alone, before and after Stage 2. Depth was the weaker single model in all six backbones.

| backbone | Depth alone (Stage 1) | change after Stage 2 | IR alone (Stage 1) | change after Stage 2 |
|---|--:|--:|--:|--:|
| r2plus1d_18 | 53.7 | +3.5 | 60.5 | −1.6 |
| r3d_18 | 55.4 | +2.4 | 57.6 | −1.8 |
| mc3_18 | 45.7 | +2.4 | 50.5 | −4.6 |
| s3d | 40.6 | +11.0 | 45.5 | +3.2 |
| mvit_v2_s | 65.4 | +5.7 | 74.4 | −2.1 |
| swin3d_t | 55.3 | −0.2 | 59.7 | −2.9 |
| **mean** | | **+4.1** | | **−1.6** |

The weaker branch (Depth) rises in 5 of 6 backbones (mean +4.1); the stronger branch (IR) falls in 5 of 6 (mean −1.6). That looks like the gate moving information from the strong sensor to the weak one. But averaging alone already does something similar: with no gate, the Stage 1 average sits +4.9 points above the weaker model and −0.45 against the stronger one. The shift is also confounded by head retraining and by picking the best epoch on the validation set. Control A is meant to decide how much of it is the gate.

For `mc3_18` the averaged output is below its own best single branch at all three stages. Averaging a wrong model with a right one can lose to the right one alone. That is a property of averaging, not of the gate.

## What this does not show

- **The baseline is the plain average, the weakest late fusion.** Against the best of six late-fusion rules at Stage 1 (average, product of experts, confidence weighting, most confident branch, learned weight, learned logistic regression), the end-to-end gate is +0.6 points on average and positive on 3 of 6 backbones ([results/vs_best_late_fusion.csv](results/vs_best_late_fusion.csv)). That baseline is the maximum over six rules chosen on the same 628 clips, so it is biased in its own favour, but it shows the gain over a tuned late fusion is small.
- **The end-to-end gain is not yet credited to the gate.** Stage 3 also gives the backbones 25 more epochs of training. Control B is meant to separate the two.
- **Every checkpoint was chosen by accuracy on the same 628 clips it is reported on.** This biases every stage upward by an unknown amount.
- **One run per configuration, and the original runs were not seeded.** They cannot be repeated exactly.
- **The mechanism is untested.** We have not tested why Depth is the weaker branch, nor whether information really moves between branches.
- **An earlier summary sent to the organisers described an asymmetric gain from the gate that was in fact an averaging effect.** The correction is in [docs/how-it-developed.md](docs/how-it-developed.md#a-correction).

## Where things stand

As of 8 October 2026. Failed and inconclusive runs stay on this table.

| experiment | what it decides | status |
|---|---|---|
| six backbones, three stages | the numbers above | done, one run each |
| Control A: Stage 2 with the gate frozen at zero | is the Stage 1 → 2 shift caused by the gate, or by head retraining and epoch selection | code written ([experiments/01_nogate_control](experiments/01_nogate_control/)); no results recorded here yet |
| Control B: end to end with the gate frozen at zero | is the Stage 3 gain caused by the gate, or by 25 more epochs of training | not in this repository. The author reports a run was made; no numbers are recorded here |
| Control C: linear heads on concatenated frozen features | does the gate beat early fusion on frozen features (see the PDF: it cannot represent more) | planned |
| seed-matched repeats | how much of any difference is run-to-run noise | planned |
| same-modality partner (IR+IR, Depth+Depth) | does the gain need two different sensors | planned |
| cross-term ablation | which part of the gate carries the effect | planned |

## Read further

1. **[Cross-gate hypothesis (PDF)](docs/cross-gate-hypothesis.pdf)**: the formal analysis in one document: what the gate can express, gradients, what each stage isolates, the statistics, the controls that would decide, and a draft philosophical reading.
2. **[Research plan](docs/research-plan.md)**: experiments a lab could run on the RGB hypotheses and the controls, the model and data sizes they apply to, estimated GPU hours, what each outcome would mean, and a future goal (adapter-based exchange). None run yet.
3. **[How the system developed](docs/how-it-developed.md)**: what was tried before the gate, what failed, a correction, what is still open.
4. **[The ML approach](docs/ml.md)**: data, preprocessing, models, training, every hyperparameter.
5. **[The philosophical approach](docs/philosophy.md)**: the question underneath, and how I got to it. *(draft)*
6. **[The mathematics: what we did](docs/math.md)**: the gate, the losses, the metrics and the controls, written out.
7. **[Side hypotheses: RGB and sensor structure](docs/side-hypotheses.md)**: related, not part of the result, not on the poster.

## Repository map

| path | what |
|---|---|
| `docs/` | the pages above, `cross-gate-hypothesis.pdf` (LaTeX source in `docs/hypothesis-src/`), `verify_math.py` (NumPy check of the identities) |
| `notebooks/CUHKX_gate_experiment_EN.ipynb` | the full experiment: data download, preprocessing, six backbones × three stages, analysis. Outputs stripped |
| `experiments/01_nogate_control/` | Control A code and how to read its output |
| `experiments/gpu_cost_model.py` | the GPU-hour estimates in the research plan: model size class × data size. Estimates, not measurements |
| `experiments/side_rgb/` | RGB and sensor-structure side hypotheses. Plans only |
| `results/` | numbers parsed from the notebook's own printed output: `per_stage_accuracy.csv`, `system_level_comparison.csv`, `fusion_methods.csv`, `summary_numbers.json` |

## Reproducing

1. Open the notebook in Google Colab with a GPU (the author used an A100).
2. Run the cells in order. Cell 0 sets paths, Cell 1 downloads the data from Hugging Face.
3. The control experiments run in the same session after Cell 16.

## Not in this repository

The dataset and the trained weights (about 3.6 GB for the six backbones). `.gitignore` blocks them.

---

Melih Talha Bektaş · Üsküdar University · CUHK-X Challenge Grand Finals, Small Model Track.
