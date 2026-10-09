[← README](../README.md)

# The philosophical approach

> **Draft.** "The question underneath" and "How I got here" are the story as told in the message sent to the organisers, put into the first person. "What an experiment can tell us" was drafted for the author and is not yet approved. The last section is for the author to write. Remove this box when all are done. The same reading, shorter, is Section 8 of [cross-gate-hypothesis.pdf](cross-gate-hypothesis.pdf).

## The question underneath

Inside a model's internal representation of information, what can we actually separate out, and how?

## How I got here

I started by trying to separate, inside a trained model, the representation of the person and their movement from everything else, without relying on how the model was trained to do it. My first attempt was a decoder that tried to pull the identity-related information back out after training. It failed. A rough check suggested the two were strongly entangled, though I do not count that check as a result. I then tried to enforce the separation during training with gradient reversal. That did not work either.

I then looked at the problem differently: how strongly does a representation depend on the kind of input it was trained on? There was a real, measurable coupling. My reference model had been trained on RGB, while the data I had was IR and Depth, so I asked whether that mismatch was itself the source of the problem. Isolating RGB's effect branched into too many sub-problems, so I set it aside. Then I tried the opposite, making IR and Depth resemble RGB at the input. That failed too.

The gate came after this. Instead of erasing a modality's influence or imitating it, let two models trained on different sensors feed each other, each compensating for what the other lacks. I want to be exact about what that rested on: an assumption I could not justify in advance, that IR and Depth are complementary. It was an intuition. The system produced a measurable effect, but a running system that returns numbers does not turn an assumption into a finding.

## A rule I tried to keep

Say what is measured, what is defensible, and what is a hunch, and keep them apart. This is why a [correction](how-it-developed.md#a-correction) sits next to the history, and why the open questions are listed next to the results.

## What an experiment can tell us

**A run tests a conjunction.** A positive number tests at once that the code is right, the split means what we think, training was not accidentally longer, the epoch choice was neutral, and the gate matters. It is tempting to credit the last item. The controls remove the others one at a time.

**Three levels of claim.** (1) There is an effect: the system scores above a baseline. (2) The effect is due to the component: removing it removes the effect. (3) The component works by the proposed mechanism: what one sensor lacks, the other supplies. Each level needs more than the one before. Today the evidence sits tentatively at level 1 for the end-to-end stage and below level 1 for the gate-only stage.

**Complementarity is an assumption.** The design assumes IR and depth carry different information about the same action. The mathematics shows the architecture can test only an additive version of that, and that giving each branch both sensors can reduce the diversity between branches, the opposite of complementarity at the level of errors. A hypothesis that names its own failure (a gate no better than a concatenation head; a gain that survives a same-modality partner) is better than one that can absorb any outcome.

**Separating, compensating, erasing.** The attempts to erase one factor from the representation (a decoder after training, gradient reversal during it) did not work, and that says something only under assumptions about what those methods can do. The gate takes the opposite stance: instead of removing what a sensor lacks, let another sensor compensate. In the simplest case that is a linear reweighting of one branch by the other. Whether a deeper form of compensation exists, one no concatenation head could imitate, is what the end-to-end stage and Control B are meant to ask.

**Keeping the record honest.** Negative and inconclusive runs sit on the same table as positive ones, and a correction to an earlier summary stays beside the history.

## To write, in the author's own words

- What I believed before the first experiment, and which beliefs the experiments overturned.
- Why separate or compensate rather than erase.
- What I would need to see to say the gate does what I first thought it did.
