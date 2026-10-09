[← README](../README.md)

# Cross-modal gating for IR and Depth action recognition: what I know, what I do not, and the experiments that would settle it

*Status: a proposal. Nothing in Sections 3 to 5 has been run. All GPU figures are estimates from counting operations, not measurements.*

## 1. Summary

**Question.** Two video models, one fed infrared (IR) and one fed depth clips of the same action, can exchange information through a small learned gate before their outputs are averaged. Does that help, and if so is it the gate or something else (extra training, retraining of the classifier, selection of the best epoch on the evaluation set)? A second, more speculative question: a model pretrained on RGB video receives IR and Depth as three identical gray channels; does that mismatch explain why Depth is the weaker sensor, and can another sensor compensate for it?

**What I have.** On the CUHK-X Small Model Track (40 actions, split by subject, 2,303 training and 628 validation clips) I ran six video backbones through three stages: independent models, gate with frozen backbones, gate trained end to end. Code, numbers and a formal analysis are in this repository.

| comparison (change in accuracy of the averaged output, points) | mean over 6 backbones | positive on |
|---|--:|--:|
| gate only, against plain average of independent models | +0.5 | 2 of 6 |
| end to end, against plain average | +2.2 | 4 of 6 |
| end to end, against the best of six late-fusion rules chosen on the same clips (a baseline biased in its favour; [table](../results/vs_best_late_fusion.csv)) | +0.6 | 3 of 6 |

The end-to-end gain has a paired *t* of 1.75 (5 degrees of freedom, p about 0.14) and one backbone carries much of it. So the honest reading is a positive tendency, not an established effect. The [formal analysis](cross-gate-hypothesis.pdf) also shows that with frozen backbones the gate cannot represent anything a pair of concatenation heads cannot, so an early-fusion control is missing, and that Depth being the weaker branch may come from how colour-mapped depth is converted to gray. An earlier summary of mine overstated an asymmetric gain; the correction is in [how-it-developed.md](how-it-developed.md).

**What I do not know.** Whether the gain survives seeds (none were set), a matched control without the gate, and epoch selection outside the evaluation clips. I ran a longer-training control without the gate earlier and it did not reproduce the gain, but its numbers were not recorded, so I treat the question as open and rerun it seeded. I have one dataset and no explanation of mechanism.

**What I propose (Section 3), in order.**

1. *Seeded controls* for the gate: no-gate control, extra-parameters control, early-fusion control, same-sensor partner, and the strongest late-fusion rules as baselines.
2. *Check how Depth is encoded*: identify the colour map of the depth frames and compare luminance conversion with an exact inversion. If Depth was weak because of this, part of the gate's gain is a repair of a preprocessing flaw.
3. *A stop-gradient test* that separates cross-learning between the models from extra capacity.

These cost about **34 A100-hours per architecture of 30M parameters** (Section 6), so two architectures fit in about three days on one A100. Then, if the first answer is yes, the RGB question (Section 4).

**What I would like from a lab.** GPU access of that order; advice on which controls would convince you; access to a dataset recording RGB, IR and Depth together (for Section 4); and a reading of my analysis by someone who will find what I missed. In return, the experiments in Section 3 run to a written result in about eight weeks, whichever way it falls.

## 2. What the data can and cannot answer

The Small Model Track data ([dataset card](https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Small_Model_Track)) contains six modalities: `Depth_Color` (colourised depth), `IR`, `Thermal`, `IMU`, `Radar`, `Skeleton`. According to the card there is **no RGB and no raw depth**, and not every clip has every modality. So:

- The RGB questions need a dataset with RGB, IR and Depth recorded together. [NTU RGB+D 120](https://arxiv.org/pdf/1905.04757) provides all three for the same 114,480 samples (120 classes, 106 subjects). Its access conditions and the licence of CUHK-X (not stated on its card) must be checked before use.
- Depth exists only as colour-mapped frames. The pipeline converts them to gray by luminance, which may not preserve depth order.

## 3. Core experiments

### Rules for every experiment

1. At least three seeds; five where the main claim is decided.
2. The kept epoch is chosen on a subject-wise split of the *training* data. The held-out subjects are used once.
3. Differences are paired on the same clips, with a bootstrap interval over clips *and* seeds.
4. Each experiment has a decision table written before the run. A result outside the table means the table was wrong, and I say so.
5. Each experiment runs on at least two architecture families of the chosen size class.
6. Failed runs stay in the record.

### 3.1 The controls (decides whether the gate matters)

Defined in the [PDF](cross-gate-hypothesis.pdf). For each architecture and seed: Stage 1; Stage 2 and Stage 3 each with the gate and with the gate frozen at zero (Controls A and B); a linear head on concatenated frozen features (Control C); a gate-sized module without exchange between the branches (extra-parameters control); and the same-sensor partner (Control D: IR with IR, Depth with Depth). Baselines include the best late-fusion rules, not only the plain average: against them the end-to-end gain shrinks to +0.6 points (table above), and that comparison must be redone with the rule chosen outside the evaluation clips.

| outcome | meaning |
|---|---|
| gate beats A in Stage 2 and beats C | additive feature mixing helps beyond head retraining and beyond early fusion on frozen features |
| gate ≈ C, both above A | the gain is feature sharing; the gate is one way to get it, and going beyond early fusion is not supported at the feature level |
| gate ≈ A in Stage 2 | head retraining and epoch selection explain Stage 2 |
| Stage 3 gate beats B and the extra-parameters control | coupling beyond extra epochs and extra parameters; go to 3.3 |
| Stage 3 gate ≈ B | the Stage 3 gain was training length |
| D shows the same gain | an ensembling or regularisation effect, not complementarity |
| gate gain disappears against the best late-fusion rule | the poster's baseline was too weak, and I say so |

### 3.2 Is Depth weak because of how it is encoded?

**Question.** In the experiments so far Depth was the weaker branch in all six backbones. The input is colour-mapped depth converted to gray by luminance. If the colour map is not monotone in luminance, near and far can look alike and information is lost before the model sees it. That would be a preprocessing flaw, not a fact about sensors.

**Step 0, no GPU.** Identify the colour map: collect pixel colours from `Depth_Color` frames, test whether they lie on a one-dimensional curve in RGB space, and compare with standard colour maps by nearest-neighbour distance. If one matches, invert it exactly; otherwise recover the ordering along the curve. Separately, inspect IR frames and the dataset paper to establish whether IR is active near-infrared (the dataset has a separate `Thermal` modality, so probably yes).

| id | input to the Depth branch | note |
|---|---|---|
| E0 | luminance gray ×3 | current |
| E1 | recovered scalar depth ×3 | exact inversion |
| E2 | E1 normalised per clip inside the person box | removes global range effects |
| E3 | three channels: depth, horizontal and vertical depth gradient | a cheap geometric encoding |
| E4 | colour as is | the earlier attempt that lowered accuracy, now with an interval |

| outcome | meaning | next |
|---|---|---|
| E1/E2/E3 clearly above E0 and Depth now close to IR | the weakness was preprocessing; part of the earlier "weaker branch rises" story is the gate repairing a flaw | rerun the gate comparison on the fixed input; a smaller gate gain would itself be the finding |
| E1/E2/E3 above E0 and the gate gain survives | stronger evidence for the gate, since the weak branch is no longer a weakness to compensate | the controls (3.1) on the fixed input |
| all within noise of E0 | the colour map was not the problem | 4.2 and the probes in Section 5 |
| E4 equals E0 | the earlier drop was noise | drop that claim |

### 3.3 What does the gate actually do?

- **Stop-gradient.** In Stage 3, compute f + A·detach(g) and g + B·detach(f). The forward pass is unchanged; the cross-gradient terms of the PDF (Bᵀ W₂ᵀ r₂ in ∂L/∂f, Aᵀ W₁ᵀ r₁ in ∂L/∂g) vanish. Three seeds, from shared Stage 1 and Stage 2.
- **Variants.** One-way A only; one-way B only; low-rank A and B (rank 16); a two-layer MLP gate (nonlinear, so it can express interaction).
- **Partner corruption at test time, no training.** Replace g by zeros, by the g of a random clip of another class, or drop a modality.

| outcome | meaning |
|---|---|
| stop-gradient removes the Stage 3 gain | the gain needs cross-gradient coupling, not just capacity |
| stop-gradient keeps the gain | capacity or training length; consistent with Control B |
| MLP gate beats linear | interaction between modalities, not only additive evidence, is useful |
| one-way gates match two-way | one direction carries the effect; state which |
| shuffled partner drops accuracy to about the single-branch level | the model uses clip-specific partner information |
| shuffled partner changes nothing | the gate acts as a learned bias, not as communication |

## 4. The RGB question, if Section 3 gives a reason to continue

**The idea as first written.** The Bayer pattern of an RGB camera samples the three channels at different positions, so RGB data may carry channel-specific spatiotemporal structure that single-channel IR or Depth lacks, and an RGB-trained model may learn something extra that IR/Depth training cannot supply. As one statement this is too broad to test, so I split it into five claims, each of which can fail:

| id | claim | decided by | what I expect |
|---|---|---|---|
| C1 | colour or channel structure carries action information in RGB beyond luminance | 4.1 | small effect |
| C2 | the mosaic (Bayer) structure specifically, not colour in general, matters | 4.1, conditions R7 and R8 | unlikely; I have found no literature for it and cite none. Commercial camera pipelines process frames before they reach a dataset, which removes most of that structure |
| C3 | three identical gray channels are off-distribution for an RGB-pretrained model, and a better input bridge helps IR and Depth | 4.2 | plausible |
| C4 | pretraining on the same sensor as the target beats pretraining on RGB | 4.3 | plausible, size unknown |
| C5 | what an RGB model knows can be passed to an IR/Depth model, by distillation or through a gate | 4.4, 4.5 | open |

C2 is the weakest claim and I treat it as a cheap side test, not as the motivation. Two earlier attempts are recorded in [how-it-developed.md](how-it-developed.md): making IR/Depth look like RGB failed, and feeding Depth in real colour lowered accuracy; neither has numbers here yet, and 3.2 and 4.2 replace them with measured results.

### 4.1 Does RGB carry more than luminance? (C1, C2)

**Data.** D2, an RGB+IR+Depth dataset. Single stream, Stage 1 only, two architectures of one size class, three seeds.

| id | RGB input | what it tests |
|---|---|---|
| R1 | RGB | reference |
| R2 | luminance gray ×3 | the treatment IR/Depth receive now |
| R3, R4, R5 | red / green / blue only, ×3 | one channel's contribution |
| R6 | chroma only (luminance fixed) | colour without brightness |
| R7 | channels permuted per clip | whether a fixed channel meaning is used |
| R8 | RGGB mosaic, then bilinear demosaic | adds known mosaic-like interpolation structure |

Plus a test-time cross-evaluation (train on R1, test on R2–R8), which costs almost nothing, and the matched reference of IR-only and Depth-only through the same architecture on the same samples.

**Limit.** Frames from such datasets come out of a camera pipeline and are compressed; the original mosaic is largely gone. R8 tests sensitivity to mosaic-like structure, not the sensor's own mosaic. A real test of C2 needs RAW sensor data, which this plan does not have.

| outcome | meaning |
|---|---|
| R1 ≈ R2 | colour adds little for this task; C1 has no support; any RGB gap is about pretraining statistics (C3), not information |
| R1 > R2, R6 above chance, R7 hurts | channel structure matters in RGB; next test whether it can be supplied (3.3, 4.4) |
| R8 ≈ R1 | no sensitivity to mosaic-like structure; C2 unsupported on processed frames |
| R8 clearly different | structure at that scale matters; RAW data needed before saying why |
| RGB far above IR and Depth on the same samples | the sensor gap is large on matched content, so the compensation question is meaningful. If small, the project's premise changes |

### 4.2 The input bridge: how should IR and Depth enter an RGB-pretrained model? (C3)

Single stream, Stage 1, on CUHK-X, IR and Depth (with the best encoding from 3.2), five conditions against the gray baseline of 3.1:

| id | condition |
|---|---|
| B1 | a learned 1×1 stem (1→3 or 3→3), initialised to replication |
| B2 | fine-tune only the stem and the first block |
| B3 | standardise to the pretraining channel statistics instead of dataset statistics |
| B4 | random initialisation, no pretraining |
| B5 | ImageNet weights inflated to 3D (ResNet-style architectures only) |

| outcome | meaning |
|---|---|
| B1–B3 above baseline | the input is off-distribution for the pretrained filters; gray replication is a poor bridge (supports C3) |
| none above baseline | the bridge is not the bottleneck |
| B4 close to baseline | pretraining on RGB is not helping IR/Depth much; with a few thousand clips this would be surprising and needs checking before believing |
| B4 far below | pretraining helps across the sensor gap; ask how much each part contributes (B2 against full fine-tuning) |

### 4.3 Does the sensor of the pretraining data matter? (C4)

Pretrain the same architecture on the same subset three times: on its RGB, on its IR, on its Depth (100 epochs each). Then fine-tune on CUHK-X IR and Depth. The comparison inside one dataset isolates the sensor, since data, actors, classes and architecture are identical. Comparing against pretraining on a different, larger dataset would confound sensor with data size and content, so no conclusion should rest on that comparison alone.

| outcome | meaning |
|---|---|
| IR-pretrained best on IR, Depth-pretrained best on Depth | pretraining sensor matters; the RGB start is a handicap |
| RGB-pretrained best everywhere | what RGB pretraining supplies is not sensor-specific (generic motion features) |
| all equal within noise | the pretraining sensor is not the factor at this scale |

### 4.4 Can an RGB model teach an IR/Depth model? (C5)

On D2. Train an RGB teacher; train IR and Depth students with and without distillation ([Gupta et al., CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Gupta_Cross_Modal_Distillation_CVPR_2016_paper.html) is the standard reference for cross-modal supervision transfer). Compare with the gate using an RGB partner (4.5).

| outcome | meaning |
|---|---|
| distilled students beat plain students | RGB-trained knowledge carries something IR/Depth training alone does not find; the "something extra" idea in a testable form |
| no gain | the extra is not transferable this way, or not there |

### 4.5 The gate with RGB as a partner (C5)

On D2, pairs RGB+IR, RGB+Depth and IR+Depth, with matched controls. If IR and Depth each lack something RGB has, the gain from an RGB partner should exceed the gain from the other low-information sensor. If gains are equal across pairs, the effect is generic (ensembling or regularisation) and not specific to what RGB carries.

## 5. Further options, not part of the first eight weeks

| option | question | cost class |
|---|---|---|
| probes and CKA | what is linearly readable in the representation (action, subject, sensor), and does the gate change it; links to my earlier attempts to separate identity from the rest ([Kornblith et al., ICML 2019](https://arxiv.org/abs/1905.00414)) | forward passes only, about 1 A100-hour at 30M |
| other sensor pairs in CUHK-X | IR+Thermal and Depth+Thermal: a pair that differs more is a harsher test of complementarity | about 3.5 A100-hours at 30M |
| other model sizes | does the finding survive in the 4M to 10M models that fit the Small Model Track's size limit, and in an 85M model | see Section 6 |
| adapter-based exchange | Section 9 | not costed |

## 6. Compute

**Model size classes** (approximate parameters and forward cost for one 16-frame clip; figures taken from published model descriptions and to be checked against the model actually used).

| class | parameters | forward cost | what falls in it |
|---|--:|--:|---|
| XS | ~4M | ~3 GFLOPs | efficient 3D CNNs built for small devices |
| S | ~10M | ~18 GFLOPs | separable-convolution 3D CNNs |
| M | ~30M | ~45 GFLOPs | ResNet-style 3D CNNs, small hierarchical video transformers |
| L | ~85M | ~180 GFLOPs | a plain ViT-Base video transformer at 224 px |

**Data sizes.** D1: 2.3k clips (the Small Model Track training split). D2: 6k (a subsample of an RGB+IR+Depth dataset, about 150 clips per class over 40 classes). D3: 38k (a 40-class subset of NTU RGB+D 120 in full, approximate). D4: 114k (NTU RGB+D 120). D5: 240k (Kinetics-400 scale).

**Unit and formula.** One *single-stream run* is fine-tuning one model for 25 epochs on one sensor stream. A gated pair trained for three stages is about 2.9 of them; with Controls A, B and C about 9.5. Script: [experiments/gpu_cost_model.py](../experiments/gpu_cost_model.py).

> A100-hours = clips × epochs × 3 × forward GFLOPs × 1.4 ÷ (15 TFLOP/s) ÷ 3600

3 is forward plus backward; 1.4 adds the per-epoch evaluation passes; 15 TFLOP/s is an assumed effective fp32 throughput on an A100. The true figure could be 0.65 times this (mixed precision, fast data loading) to 1.7 times (slow loading). Data loading is not modelled.

**Table A. One single-stream 25-epoch run, A100-hours.**

| model class | D1 2.3k | D2 6k | D3 38k | D4 114k | D5 240k |
|---|--:|--:|--:|--:|--:|
| XS | 0.01 | 0.03 | 0.22 | 0.67 | 1.4 |
| S | 0.08 | 0.21 | 1.3 | 4.0 | 8.4 |
| M | 0.20 | 0.53 | 3.3 | 10 | 21 |
| L | 0.8 | 2.1 | 13 | 40 | 84 |

**Table B. Each experiment, one architecture of the class, A100-hours.** Two architectures double it (rule 5).

| experiment | data | XS | S | M | L |
|---|---|--:|--:|--:|--:|
| 3.1 controls A, B, C, extra-parameters, 5 seeds | D1 | 0.6 | 3.8 | 9.6 | 38 |
| 3.1 control D, same-sensor partner, 3 seeds | D1 | 0.5 | 2.8 | 6.9 | 28 |
| 3.2 depth encodings, 5 × 3 seeds, then the gate | D1 | 0.4 | 2.6 | 6.5 | 26 |
| 3.3 stop-gradient in Stage 3, 3 seeds | D1 | 0.1 | 0.6 | 1.6 | 6.2 |
| 3.3 variants and partner corruption | D1 | 0.6 | 3.8 | 9.6 | 38 |
| 4.1 RGB ladder, 8 conditions × 3 seeds | D2 | 0.8 | 5.0 | 13 | 50 |
| 4.2 input bridge, 2 sensors × 5 conditions × 3 seeds | D1 | 0.4 | 2.4 | 6.0 | 24 |
| 4.3 pretraining by sensor (3 runs of 100 epochs) and 18 fine-tunes | D3 + D1 | 2.9 | 17 | 44 | 174 |
| 4.4 RGB teacher to IR/Depth students | D2 | 0.3 | 1.8 | 4.5 | 18 |
| 4.5 the gate with RGB as partner, 3 pairs × 3 seeds | D2 | 1.8 | 11 | 27 | 108 |
| 5 probes and similarity (forward passes only) | D1 | 0.1 | 0.5 | 1.2 | 4.6 |
| 5 other sensor pairs inside CUHK-X | D1 | 0.2 | 1.4 | 3.5 | 14 |

**Table C. Totals, one architecture per size class, A100-hours.**

| part | XS | S | M | L |
|---|--:|--:|--:|--:|
| Section 3, core experiments | 2.3 | 14 | 34 | 136 |
| Section 4, RGB question | 6.2 | 37 | 94 | 375 |
| Section 5, further options | 0.3 | 1.8 | 4.6 | 18 |
| all | 8.8 | 53 | 132 | 530 |

How to read it: Section 3 on an M-class architecture is about 34 A100-hours; on a node with eight A100s that is an afternoon, not counting data preparation. The cost is dominated by the L class and by the pretraining and RGB-partner experiments, not by the controls. Not included: data preparation and person cropping, debugging, failed runs, hyper-parameter search, and self-supervised pretraining from scratch (hundreds of epochs on D4 or D5, a different cost class, out of scope).

## 7. Schedule for an eight-week stay (adjust to the lab)

| weeks | work |
|---|---|
| 1–2 | identify the colour map of `Depth_Color` (no GPU); add seeds and the inner validation split to the training code; check data access terms; prepare the second dataset in parallel |
| 3–4 | 3.1 controls and 3.2 depth encodings on two M-class architectures (about 46 A100-hours) |
| 5–6 | 3.3 stop-gradient and variants (about 22 A100-hours); 4.1 RGB ladder if the data is available |
| 7–8 | analysis, intervals, write-up of the result whatever its sign; 4.2 or 4.4 if time allows |

Deliverable: a short report and a clean repository in which every number is seeded, selected outside the evaluation set, and comes with its control. A negative result is a deliverable.

## 8. Limits and risks

- **The evidence so far is weak.** See Section 1. The plan exists because of that.
- **One dataset, four held-out subjects.** One accuracy on 628 clips has a standard error of about 2 points.
- **Dataset terms.** Gated access on CUHK-X, licence not stated; the other dataset's conditions not checked. Weights and data stay out of the repository.
- **A second dataset is not CUHK-X.** Conclusions from Section 4 are about that dataset; transfer to CUHK-X is a hypothesis.
- **The estimates.** The hours are a guess within a factor of about two until measured.
- **Scope.** Section 3 alone answers the first question; Section 4 is conditional on it.

## 9. Future goal: from a linear gate to adapter-based exchange between models

> Not implemented. Nothing in this section has been run, and the current system does not contain what is described here.

**What the current system exchanges.** One model passes the other a linear function of its *feature vector* (an activation): f̃ = f + A g, with A a full d × d matrix. It does not pass weights, weight updates or low-rank adapters, and A is not low-rank. The distinction matters: exchange in activation space changes what a branch sees at inference; exchange in weight space would change how a branch is trained or adapted.

**The goal.** Test whether *low-rank adapters* (LoRA-style modules in intermediate layers) whose input or modulation comes from the partner model give a better cost-to-benefit than a last-layer linear gate, while each model keeps its own specialisation. A reader would reasonably first ask whether any gain is only extra parameters, and then whether it is the exchange.

**The comparison ladder.** Same data splits, same seeds, matched trainable parameters and A100-hours reported, epochs chosen outside the reported set, training-length controls as in 3.1.

| row | system | what the comparison isolates |
|---|---|---|
| L0 | plain late fusion | reference |
| L1 | late fusion + independent adapters in each model, no exchange | is the gain just extra parameters |
| L2 | late fusion + the current linear gate | what today's gate adds (this is the system on the poster) |
| L3 | late fusion + mutual low-rank adapter exchange | the contribution of sharing information through adapters |
| L4 | gate + mutual adapters | whether the two mechanisms add up |

| outcome | meaning |
|---|---|
| L1 ≈ L2 ≈ L3 above L0 | extra capacity explains it; the exchange is not what helps |
| L3 above L1 and L2 | information exchange through adapters carries a gain beyond parameters and beyond the linear gate |
| L2 above L1 and L3 | the cheap linear gate is the better design |
| L4 above all | the mechanisms are complementary |

**Prior work to compare against.** Verified to exist while writing this plan:

- [UniAdapter](https://arxiv.org/pdf/2302.06605), ICLR 2024: adapters on each modality and on their cross-modal interactions in vision-language models, with partial weight sharing so that the modalities share knowledge. Closest in spirit.
- [D-MoLE](https://proceedings.mlr.press/v267/ge25d.html), ICML 2025: LoRA experts for continual instruction tuning of a *single* multimodal language model, with updates adjusted by how hard each modality is. It is about continual learning, not about two models exchanging information, so it is relevant only for the LoRA-expert idea.
- A paper called MCDGait, said to use two-way information transfer between modality networks: **I could not find it**, so it is not cited here. It must be located and read before it is used.

LoRA, gating and two-way transfer are each established ideas in some form, and feature-exchange modules between modality-specific networks exist as well (cross-stitch units and MMTM, named from memory and not checked here). Any novelty would be in the specific combination and in the evidence for it, not in the pieces.

## 10. References checked while writing this plan

- Kornblith, Norouzi, Lee, Hinton, *Similarity of Neural Network Representations Revisited*, ICML 2019. [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)
- Gupta, Hoffman, Malik, *Cross Modal Distillation for Supervision Transfer*, CVPR 2016. [paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Gupta_Cross_Modal_Distillation_CVPR_2016_paper.html)
- Wang, Tran, Feiszli, *What Makes Training Multi-Modal Classification Networks Hard?*, CVPR 2020. [paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.html)
- Liu et al., *NTU RGB+D 120: A Large-Scale Benchmark for 3D Human Activity Understanding*. [arXiv:1905.04757](https://arxiv.org/pdf/1905.04757)
- Eitel et al., *Multimodal Deep Learning for Robust RGB-D Object Recognition*, 2015, proposes an encoding of depth for use with pretrained CNNs. [arXiv:1507.06821](https://arxiv.org/abs/1507.06821v2). Only the abstract was read; check the encoding in the paper before citing it for a specific method.
- Lu et al., *UniAdapter*, ICLR 2024, and Ge et al., *D-MoLE*, ICML 2025 (links in Section 9).

Earlier references on modality competition and information-bottleneck claims are in [how-it-developed.md](how-it-developed.md).
