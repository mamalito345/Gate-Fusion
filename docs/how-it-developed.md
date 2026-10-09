[← README](../README.md)

# How the system developed

## The question

Take two video models. One sees an infrared stream, the other a depth stream of the same person doing one of 40 everyday actions. The standard way to combine them is to run both and average their output probabilities. We tested a different way: before each model classifies, it receives a learned linear projection of the other model's feature vector, added to its own. Then the two outputs are averaged as before. The system and the numbers are on the [README](../README.md); the details are in [the ML approach](ml.md) and [the mathematics](math.md).

## How it developed

This is the author's account, in order. The decoder, gradient-reversal and RGB-related attempts below are from the author's notes. Their code is not in this repository, so treat them as reports, not as results.

| attempt | outcome | evidence here |
|---|---|---|
| A decoder to pull identity information back out of a trained model | failed. An exploratory check suggested identity and the rest were strongly entangled, but the check was not rigorous enough to count as a result | notes only |
| Gradient reversal against subject identity during training | did not succeed | notes only |
| Post-hoc LEACE erasure of subject information, IR only, 3 seeds | a subject probe fell from about 77% to about 0.2%, but cross-subject accuracy did not change reliably (mean difference +0.05, spread ±1.22 points) | notes only |
| Isolating the effect of the RGB-trained starting point | branched into too many sub-problems; set aside | see [side hypotheses](side-hypotheses.md) |
| Making IR and Depth look like RGB at the input | failed | notes only |
| Giving the Depth branch real colour instead of gray copied to three channels | accuracy dropped, so gray was kept | numbers not yet in this repository |
| A first MMTM fusion of IR and Depth | the gate collapsed to a constant. Fixed after re-reading the original paper: an L2 penalty on the excitation signal, applied only to the last two residual blocks (layer3 and layer4) | notes only |
| Combining a large teacher ensemble | a 94–96% oracle was a ceiling, never an achieved result. Real fusion of the same models reached about 78–80% | notes only |
| Gate fusion, six backbones, three stages | a measurable end-to-end effect whose cause is not established | [notebook](../notebooks/CUHKX_gate_experiment_EN.ipynb), [results](../results/) |

The gate idea came after these failures: instead of erasing a modality's influence or imitating RGB, let two models trained on different sensors compensate for each other. It rested on an assumption that was never justified in advance, that IR and Depth are complementary. The system produced a measurable effect. An unproven assumption does not become a confirmed one because the code runs and returns numbers.

## A correction

An earlier summary of this work, sent to the organisers before the event, reported an asymmetric gain from the gate. At Stage 2, the averaged output was about +2.4 points above the weaker branch and +0.45 above the stronger one, higher than the stronger branch in 5 of 6 backbones (the exception was `mc3_18`, −0.96).

That comparison was wrong as a measure of the gate. Both reference points were the gated branches of the same stage, and the quantity being compared was their own average, so the difference measures averaging. The same calculation at Stage 1, where there is no gate at all, gives +4.9 and −0.45. The comparison on the README, gate then average against a plain average of independent models, replaces it.

## What is still open

- **Controls A and B, then seed-matched repeats.** The original runs were not seeded, so they cannot be repeated exactly.
- **Why Depth is the weaker branch in every backbone.** One reason to check: the Depth frames are colour-mapped, and the pipeline converts them to gray by luminance, which may not keep depth order. That depends on the colour map and has not been checked.
- **The mechanism.** The literature on multimodal training describes a stronger modality dominating a weaker one (for example Peng et al., CVPR 2022, on-the-fly gradient modulation; Huang et al., ICML 2022, on modality competition). The claim that a network "discards" information early and a partner restores it is disputed (Saxe et al., ICLR 2018). We have tested neither.
