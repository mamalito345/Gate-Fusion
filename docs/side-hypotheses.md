[← README](../README.md)

# Side hypotheses: RGB and sensor structure

**Outside the poster.** These are related to the main work but do not change its result. They are kept here, apart from it, with their status stated.

## The hypothesis as first written

> This may relate to how RGB is actually captured. In many Bayer-pattern camera pipelines, the red, green and blue channels respond to different wavelengths and are sampled at different physical sensor positions before demosaicing, which may introduce channel-specific spatiotemporal structure that a single-channel IR or Depth input simply does not have. If that is right, an RGB-trained model may be implicitly learning something extra alongside motion that IR/Depth training cannot supply an equivalent for.

## What the code says about the inputs

In this pipeline both sensors enter every backbone as three identical channels. IR is read as gray. Depth_Color holds colour-mapped depth frames; they are read in colour, converted to gray by luminance, and copied into three channels. So:

- Neither input carries channel-specific structure of any kind, Bayer-related or not.
- Both branches get the same treatment, so the hypothesis cannot, by itself, explain why one branch behaves differently from the other. Depth is the weaker branch in all six backbones.
- Any difference between the three channels of the first convolution's filters would come from the Kinetics-400 pretraining on RGB video, not from this data.

This is reasoning from the code. No experiment here tests it. My reading of the literature is that it gives little support to a Bayer-pattern effect on learned video features. I will add checked citations before treating that as more than an impression.

> **Update, 9 October.** The Small Model Track dataset card lists no RGB and no raw depth, so the RGB claims cannot be tested on it. The [research plan](research-plan.md) splits this hypothesis into five claims and moves the RGB tests to a dataset with matched RGB, IR and Depth.

## What has been tried

Feeding Depth as real RGB instead of gray copied to three channels: the author reports accuracy dropped, so gray was kept. The numbers are not in this repository yet.

## What is planned

| experiment | question | status |
|---|---|---|
| Real-colour Depth on one backbone (`mvit_v2_s`), Stage 1 Depth only | is the drop real, and how large, with a number and an interval | planned |
| Channel weights of the first convolution, across the six backbones | is there a channel preference at all, given identical input channels. Expected: inherited from pretraining | planned. It is a weight statistic, not proof of use |
| Colour-drift score per clip against per-clip Depth errors | does colour shift during motion predict where the Depth branch fails | planned |
| Minimal initialisation ablation | does the pretrained RGB starting point matter at all for IR and Depth | planned, low priority |

## Two things not yet checked

- Whether converting colour-mapped depth to gray by luminance keeps depth order. It depends on the colour map.
- Whether the dataset's "IR" is active near-infrared or thermal. This matters for any argument about what each sensor captures.
