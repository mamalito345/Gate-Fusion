[← README](../README.md)

# The ML approach

## Data

- CUHK-X Small Model Track training split, from [Hugging Face](https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Small_Model_Track). Only two of its six modalities are used: IR and Depth_Color.
- 40 action classes. Each clip is a folder of individual frames per modality, not a video file.
- Split by person: 2,303 training clips; 628 validation clips from four held-out subjects (users 1, 8, 9, 21). Every number in this repository is on those 628 clips. The official test set is not used in this study.

## Preprocessing

- Frame order comes from the frame index in each file name, not from sorting names alphabetically.
- A YOLOv8x person detector runs on the first 8 frames of the IR clip. The union of the boxes, with a margin factor of 1.5, is the crop. The same box is used for Depth, because the two cameras are synchronised and aligned.
- 24 frames are sampled evenly from each clip and cached at 160×160. Training uses 16 of them.
- IR is read as gray. Depth_Color is read in colour, then converted to gray by luminance. Both are then copied into three identical channels, so every backbone sees the same kind of input for both sensors.
- Each backbone is resized to the resolution it was pretrained at: 112 for the R-family, 224 for S3D, MViT and Swin3D.
- Training augmentation: 3 of the 16 frames are zeroed at random. Nothing else.

## Models

Six torchvision video backbones, all with Kinetics-400 weights: `r2plus1d_18`, `r3d_18`, `mc3_18`, `s3d`, `mvit_v2_s`, `swin3d_t`. Each system uses two copies of the same backbone, one per sensor, plus a linear classification head on each. The gate is described on the [mathematics page](math.md).

## Training

- Three stages of 25 epochs: independent models, then gate and heads with frozen backbones, then everything together. See the [README](../README.md#the-system).
- AdamW with PyTorch defaults, linear warm-up then cosine decay, cross-entropy per branch, no weight-decay override, no gradient clipping, no mixed precision.
- Stage 2 trains the gates and heads at the gate learning rate. Stage 3 trains every parameter, gates and heads included, at the backbone learning rate.
- The kept epoch is the one with the highest mean of the two branches' accuracy on the validation set. That is the same set the results are reported on. It is a known optimism and applies to every stage equally.
- If the GPU runs out of memory the batch size is halved and that backbone's training restarts.
- **No random seed is set anywhere in the original notebook.** The runs are one-off and cannot be reproduced exactly.

| backbone | input size | batch | backbone lr | gate and heads lr (Stage 2) | warm-up fraction |
|---|--:|--:|--:|--:|--:|
| r2plus1d_18 | 112 | 64 | 1e-4 | 2e-3 | 0.05 |
| r3d_18 | 112 | 64 | 1e-4 | 2e-3 | 0.05 |
| mc3_18 | 112 | 64 | 1e-4 | 2e-3 | 0.05 |
| s3d | 224 | 32 | 1e-4 | 2e-3 | 0.05 |
| mvit_v2_s | 224 | 16 | 5e-5 | 2e-3 | 0.10 |
| swin3d_t | 224 | 24 | 5e-5 | 2e-3 | 0.10 |

## Other ways of fusing the outputs

Besides the plain average, the notebook compares product of experts, confidence weighting, picking the more confident branch, a learned mixing weight, and a learned logistic regression, all with leave-one-subject-out evaluation. No method wins consistently across backbones and stages. Counting which one scored highest in each of the 18 cells: product of experts (7), learned alpha (loso) (6), confidence weighted (3), plain average (2). Because the winner is picked on the same 628 clips, those counts are optimistic. Full table: [results/fusion_methods.csv](../results/fusion_methods.csv).

## Reproducing it

1. Open [notebooks/CUHKX_gate_experiment_EN.ipynb](../notebooks/CUHKX_gate_experiment_EN.ipynb) in Colab with a GPU. The author used an A100.
2. Run the cells in order. Cell 0 sets paths and Cell 1 downloads the data from Hugging Face.
3. The control experiments are in [experiments/01_nogate_control/](../experiments/01_nogate_control/) and run in the same session after Cell 16.
4. Model weights (about 3.6 GB for the six backbones) are not in the repository.
