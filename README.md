# UltraRefiner: End-to-End Differentiable Segmentation Refinement

## Abstract

Medical image segmentation, particularly for breast lesions in ultrasound, remains challenging due to low contrast, speckle noise, and ambiguous boundaries. While deep learning methods like TransUNet achieve reasonable performance, they often produce masks with imprecise boundaries, topological errors, or complete failures on difficult cases. The Segment Anything Model (SAM) offers powerful refinement capabilities but requires carefully designed prompts and cannot be trained end-to-end with upstream segmentation networks.

**UltraRefiner** addresses these limitations by introducing a **fully differentiable pipeline** that connects a coarse segmentation network (TransUNet) with SAM for mask refinement. The key innovation is that **gradients flow from SAM's output back through the prompt generation to TransUNet**, enabling joint optimization of both networks. This is achieved through three differentiable prompt extraction mechanisms:

1. **Point Prompts via Soft-Argmax**: The foreground centroid is computed as a probability-weighted average of coordinates, and a background point is extracted from the inverse mask within the bounding box region.

2. **Box Prompts via Weighted Statistics**: Bounding boxes are computed using the mask-weighted mean and standard deviation of coordinates (center ± 2.5σ), ensuring the box tightly covers the predicted region.

3. **Mask Prompts via Direct/Gaussian Conversion**: The soft probability mask is converted to logit space and optionally smoothed with Gaussian blur to match SAM's expected input distribution.

The training follows a **three-phase strategy**:
- **Phase 1**: Train TransUNet independently on each dataset
- **Phase 2**: Finetune SAM using synthetic failure simulation (SDF-based augmentation) to learn refinement from diverse error patterns
- **Phase 3**: End-to-end joint optimization with gradients flowing through differentiable prompts

Key technical contributions include:
- **Differentiable prompt generation** enabling true end-to-end learning
- **SDF-based failure simulation** creating realistic segmentation errors for SAM training
- **Quality-aware loss** that penalizes unnecessary modifications to already-good predictions
- **Phase-consistent distribution matching** ensuring SAM sees identical input distributions during finetuning and end-to-end training

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         UltraRefiner Pipeline                             │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   Input Image (224×224)                                                   │
│         │                                                                 │
│         ▼                                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐    │
│   │                    TransUNet                                     │    │
│   │   ResNet50 ──► ViT-B/16 Transformer ──► CNN Decoder + Skip      │    │
│   └─────────────────────────────┬───────────────────────────────────┘    │
│                                 │                                         │
│                                 ▼                                         │
│                    Coarse Mask (Soft Probability)                         │
│                         P(lesion) ∈ [0, 1]                                │
│                                 │                                         │
│              ┌──────────────────┼──────────────────┐                     │
│              │                  │                  │                      │
│              ▼                  ▼                  ▼                      │
│        ┌──────────┐      ┌───────────┐      ┌──────────┐                 │
│        │  Points  │      │    Box    │      │   Mask   │                 │
│        │ soft-    │      │ weighted  │      │ gaussian │                 │
│        │ argmax   │      │ mean±std  │      │ /direct  │                 │
│        └────┬─────┘      └─────┬─────┘      └────┬─────┘                 │
│             │                  │                  │                       │
│             └──────────────────┼──────────────────┘                      │
│                                │                                          │
│                    Resize to 1024×1024                                    │
│                                │                                          │
│                                ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────┐    │
│   │                      SAM Refiner                                 │    │
│   │   Image Encoder ──► Prompt Encoder ──► Mask Decoder             │    │
│   │     (frozen)         (trainable)        (trainable)             │    │
│   └─────────────────────────────┬───────────────────────────────────┘    │
│                                 │                                         │
│                                 ▼                                         │
│                         Refined Mask                                      │
│                                                                           │
│   ◄─── Gradient Flow: L_refined → SAM → Prompts → TransUNet ───►         │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

## Differentiable Prompt Generation

The core innovation enabling end-to-end training is **fully differentiable prompt extraction** from soft probability masks. All three prompt types maintain gradient flow.

### Point Prompts (Soft-Argmax)

**Positive Point** - Foreground centroid via probability-weighted average:
```
x_center = Σ(P(i,j) × x_j) / Σ(P(i,j))
y_center = Σ(P(i,j) × y_i) / Σ(P(i,j))
```

**Negative Point** - Background centroid within the bounding box:
```
inv_mask = (1 - P) × soft_box_mask
x_neg = Σ(inv_mask × x) / Σ(inv_mask)
y_neg = Σ(inv_mask × y) / Σ(inv_mask)
```

The soft box mask uses sigmoid boundaries for differentiability.

### Box Prompts (Weighted Statistics)

Bounding boxes are computed from mask-weighted coordinate statistics:
```
center_x = Σ(P × x) / Σ(P)           # Weighted centroid
center_y = Σ(P × y) / Σ(P)

std_x = √(Σ(P × (x - center_x)²) / Σ(P))   # Weighted std
std_y = √(Σ(P × (y - center_y)²) / Σ(P))

x1, x2 = center_x ∓ 2.5 × std_x     # Box bounds (covers ~99%)
y1, y2 = center_y ∓ 2.5 × std_y
```

This approach correctly restricts computation to the mask region, unlike soft-min/max which can be dominated by background pixels.

### Mask Prompts

The soft probability mask is converted to SAM's expected logit format:
```
mask_logits = (P × 2 - 1) × 10      # Maps [0,1] → [-10, +10]
```

**Gaussian style** (default for unfinetuned SAM): Applies Gaussian blur before conversion to create softer boundaries matching SAM's training distribution.

**Direct style** (for Phase 2-finetuned SAM): No blur, preserves TransUNet's output distribution.

### Phase 2 vs Phase 3 Prompt Generation

| Aspect | Phase 2 (SAM Finetuning) | Phase 3 (E2E Training) |
|--------|--------------------------|------------------------|
| **Input Mask** | Augmented GT (soft, Gaussian-blurred) | TransUNet output (soft probability) |
| **Mask Resolution** | 224→1024 (simulates TransUNet path) | 224→1024 (actual TransUNet output) |
| **Point Extraction** | `DifferentiableSAMRefiner.extract_soft_points()` | Same code |
| **Box Extraction** | `DifferentiableSAMRefiner.extract_soft_box()` | Same code |
| **Mask Preparation** | `prepare_mask_input(style='direct'/'gaussian')` | Same code, same style |
| **Coordinate Space** | 1024×1024 pixels | 1024×1024 pixels |

**Critical**: Phase 2 and Phase 3 must use identical `mask_prompt_style` for distribution consistency.

## Training Pipeline

### Phase 1: TransUNet Training
Train coarse segmentation independently on each dataset with 5-fold cross-validation.

```bash
python scripts/train_transunet.py \
    --data_root ./dataset/processed \
    --dataset BUSI \
    --fold 0 \
    --vit_pretrained ./pretrained/R50+ViT-B_16.npz \
    --max_epochs 150
```

### Phase 2: SAM Finetuning with Failure Simulation

Generate augmented data with synthetic segmentation failures, then finetune SAM:

```bash
# Generate augmented data (soft masks for Phase 3 compatibility)
python scripts/generate_augmented_data.py \
    --data_root ./dataset/processed \
    --output_dir ./dataset/augmented_soft \
    --target_samples 100000 \
    --use_sdf \
    --soft_masks

# Finetune SAM
python scripts/finetune_sam_augmented.py \
    --data_root ./dataset/augmented_soft \
    --dataset COMBINED \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --mask_prompt_style direct \
    --transunet_img_size 224 \
    --epochs 50
```

**Failure Simulation**: SDF-based augmentation creates realistic errors (erosion, dilation, holes, bridges) with controlled Dice distribution (no perfect masks).

### Phase 3: End-to-End Training

Joint optimization with gradients flowing through differentiable prompts:

```bash
# With Phase 2-finetuned SAM
python scripts/train_e2e.py \
    --data_root ./dataset/processed \
    --datasets BUSI \
    --fold 0 \
    --transunet_checkpoint ./checkpoints/transunet/BUSI/fold_0/best.pth \
    --sam_checkpoint ./checkpoints/sam_finetuned/COMBINED/best_sam.pth \
    --mask_prompt_style direct \
    --max_epochs 100

# Without Phase 2 (unfinetuned SAM) - use gaussian style
python scripts/train_e2e.py \
    --data_root ./dataset/processed \
    --datasets BUSI \
    --fold 0 \
    --transunet_checkpoint ./checkpoints/transunet/BUSI/fold_0/best.pth \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --mask_prompt_style gaussian \
    --max_epochs 100
```

### Mask Prompt Style Selection

| SAM Checkpoint | Recommended Style | Reason |
|----------------|-------------------|--------|
| Unfinetuned (MedSAM/SAM) | `gaussian` | Softens boundaries to match SAM's training |
| Phase 2 finetuned (soft masks) | `direct` | Maintains distribution consistency |

## ROI Cropping Mode (Optional)

Focuses SAM computation on the lesion region at full 1024×1024 resolution:

```bash
# Phase 2 with ROI
python scripts/finetune_sam_augmented.py \
    --use_roi_crop --roi_expand_ratio 0.2 ...

# Phase 3 with ROI (must match Phase 2)
python scripts/train_e2e.py \
    --use_roi_crop --roi_expand_ratio 0.2 ...
```

All ROI operations (crop, paste, box extraction) are fully differentiable via `grid_sample`.

## Project Structure

```
UltraRefiner/
├── models/
│   ├── sam_refiner.py        # Differentiable SAM Refiner + ROI Cropper
│   ├── ultra_refiner.py      # End-to-end model
│   └── transunet/            # TransUNet backbone
├── scripts/
│   ├── train_transunet.py    # Phase 1
│   ├── generate_augmented_data.py
│   ├── finetune_sam_augmented.py  # Phase 2
│   └── train_e2e.py          # Phase 3
├── data/
│   └── dataset.py            # K-fold data loaders
└── utils/
    └── losses.py             # Dice, BCE, quality-aware losses
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.0

```bash
pip install -r requirements.txt

# Download pretrained weights
wget https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz -P ./pretrained/
# Download MedSAM from https://github.com/bowang-lab/MedSAM
```

## Citation

```bibtex
@article{ultrarefiner2024,
  title={UltraRefiner: End-to-End Differentiable Segmentation Refinement},
  author={},
  year={2024}
}
```

## References

- [TransUNet](https://arxiv.org/abs/2102.04306)
- [MedSAM](https://github.com/bowang-lab/MedSAM)
- [SAM](https://segment-anything.com/)
