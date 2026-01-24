# UltraRefiner: End-to-End Differentiable Segmentation Refinement

## Abstract

Medical image segmentation, particularly for breast lesions in ultrasound, remains challenging due to low contrast, speckle noise, and ambiguous boundaries. While deep learning methods like TransUNet achieve reasonable performance, they often produce masks with imprecise boundaries, topological errors, or complete failures on difficult cases. The Segment Anything Model (SAM) offers powerful refinement capabilities but requires carefully designed prompts and cannot be trained end-to-end with upstream segmentation networks.

**UltraRefiner** addresses these limitations by introducing a **fully differentiable pipeline** that connects a coarse segmentation network (TransUNet) with SAM for mask refinement. The key innovation is that **gradients flow from SAM's output back through the prompt generation to TransUNet**, enabling joint optimization of both networks. This is achieved through three differentiable prompt extraction mechanisms:

1. **Point Prompts via Soft-Argmax**: The foreground centroid is computed as a probability-weighted average of coordinates, and a background point is extracted from the inverse mask within the bounding box region.

2. **Box Prompts via Weighted Statistics**: Bounding boxes are computed using the mask-weighted mean and standard deviation of coordinates (center ± 2.5σ), ensuring the box tightly covers the predicted region.

3. **Mask Prompts via Direct/Gaussian Conversion**: The soft probability mask is converted to logit space and optionally smoothed with Gaussian blur to match SAM's expected input distribution.

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

## Training Pipeline

The training follows a **three-phase strategy**:

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

---

### Phase 2: SAM Finetuning (Critical for Phase 3 Alignment)

Phase 2 prepares SAM to work with TransUNet's output distribution. **The key is ensuring Phase 2 sees the SAME input distribution as Phase 3.**

#### Why Phase 2 Matters

In Phase 3, TransUNet produces **soft probability masks** (values in [0,1] with smooth boundaries). If SAM was only trained on binary masks, it will struggle with soft inputs. Phase 2 bridges this gap by:

1. **Generating soft masks** that mimic TransUNet's output distribution
2. **Simulating the resolution path** (224→1024) that occurs in Phase 3
3. **Teaching SAM when to refine and when to preserve** via quality-aware loss

#### Step 2.1: Generate Augmented Data

Generate synthetic "coarse masks" with controlled failure patterns:

```bash
python scripts/generate_augmented_data.py \
    --data_root ./dataset/processed \
    --output_dir ./dataset/augmented_soft \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --target_samples 100000 \
    --soft_masks \
    --use_sdf
```

**Key flags explained:**

| Flag | Purpose |
|------|---------|
| `--soft_masks` | **CRITICAL**: Generates soft probability maps (NPY float) instead of binary masks. These have Gaussian-blurred boundaries matching TransUNet's output distribution. |
| `--use_sdf` | Uses SDF-based augmentation for smoother, more anatomically plausible deformations |
| `--target_samples 100000` | Creates 100K augmented samples from your original data |

**Generated data distribution (NO Dice=1.0 perfect masks):**
- 25% with Dice 0.9-0.99 (minor artifacts, should be preserved)
- 40% with Dice 0.8-0.9 (moderate errors, needs refinement)
- 35% with Dice 0.6-0.8 (severe failures, needs strong correction)

#### Step 2.2: Finetune SAM

```bash
python scripts/finetune_sam_augmented.py \
    --data_root ./dataset/augmented_soft \
    --dataset COMBINED \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --output_dir ./checkpoints/sam_finetuned \
    --mask_prompt_style direct \
    --transunet_img_size 224 \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4
```

**Key flags for Phase 3 alignment:**

| Flag | Value | Purpose |
|------|-------|---------|
| `--mask_prompt_style` | `direct` | Since soft masks already have smooth boundaries, no extra blur needed. **Must match Phase 3 setting.** |
| `--transunet_img_size` | `224` | **CRITICAL**: Simulates Phase 3 resolution path. Coarse masks are first resized to 224×224, then upscaled to 1024×1024. This bilinear interpolation creates additional smoothing that Phase 3 naturally has. |

**What SAM learns in Phase 2:**
1. Refine boundaries from soft probability inputs
2. Fix common segmentation failures (erosion, dilation, holes, bridges)
3. **Preserve good inputs** - quality-aware loss penalizes changes to high-quality masks

#### Phase 2 Output

Two checkpoints are saved:
- `best.pth` - Full DifferentiableSAMRefiner state (for resume)
- `best_sam.pth` - SAM-compatible checkpoint (for Phase 3)

---

### Phase 3: End-to-End Training

Joint optimization with gradients flowing through differentiable prompts.

#### Recommended Command (with TransUNet Protection)

During E2E training, gradients from SAM can destabilize TransUNet. Use these protection mechanisms:

```bash
# RECOMMENDED: Full E2E training with TransUNet protection
python scripts/train_e2e.py \
    --data_root ./dataset/processed \
    --datasets BUSI \
    --fold 0 \
    --transunet_checkpoint ./checkpoints/transunet/BUSI/fold_0/best.pth \
    --sam_checkpoint ./checkpoints/sam_finetuned/COMBINED/best_sam.pth \
    --mask_prompt_style direct \
    --max_epochs 100 \
    --transunet_lr 1e-6 \
    --sam_lr 1e-5 \
    --coarse_loss_weight 0.7 \
    --refined_loss_weight 0.3 \
    --transunet_grad_scale 0.1 \
    --transunet_weight_reg 0.01 \
    --grad_clip 1.0
```

#### TransUNet Protection Options

| Flag | Recommended | Purpose |
|------|-------------|---------|
| `--transunet_lr` | `1e-6` | Lower learning rate prevents large weight updates |
| `--coarse_loss_weight` | `0.7` | Higher weight maintains TransUNet quality |
| `--transunet_grad_scale` | `0.1` | Scales gradients from SAM to 10% (reduces SAM's influence) |
| `--transunet_weight_reg` | `0.01` | L2 penalty anchors weights to Phase 1 checkpoint |
| `--freeze_transunet_epochs` | `5-10` | Optional: let SAM adapt first before joint training |

The script automatically:
1. **Evaluates TransUNet baseline** before training starts
2. **Compares performance** at each epoch (warns if TransUNet degrades)
3. **Logs to TensorBoard** for monitoring

#### Basic Command (without protection)

```bash
python scripts/train_e2e.py \
    --data_root ./dataset/processed \
    --datasets BUSI \
    --fold 0 \
    --transunet_checkpoint ./checkpoints/transunet/BUSI/fold_0/best.pth \
    --sam_checkpoint ./checkpoints/sam_finetuned/COMBINED/best_sam.pth \
    --mask_prompt_style direct \
    --max_epochs 100
```

**Phase 2 → Phase 3 Consistency Requirements:**

| Setting | Phase 2 | Phase 3 | Why |
|---------|---------|---------|-----|
| `mask_prompt_style` | `direct` | `direct` | Same mask preprocessing |
| Resolution path | 224→1024 | 224→1024 | Same interpolation smoothing |
| `use_roi_crop` | match | match | Same cropping behavior |
| `roi_expand_ratio` | match | match | Same crop boundaries |

#### Alternative: Skip Phase 2 (Use Unfinetuned SAM)

If you skip Phase 2, use gaussian style to soften TransUNet's outputs:

```bash
python scripts/train_e2e.py \
    --data_root ./dataset/processed \
    --datasets BUSI \
    --fold 0 \
    --transunet_checkpoint ./checkpoints/transunet/BUSI/fold_0/best.pth \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --mask_prompt_style gaussian \
    --transunet_grad_scale 0.1 \
    --transunet_weight_reg 0.01 \
    --max_epochs 100
```

This approach matches the SAMRefiner paper configuration (unfinetuned SAM + gaussian + points + box).

---

## Differentiable Prompt Generation

The core innovation enabling end-to-end training is **fully differentiable prompt extraction** from soft probability masks.

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

**Gaussian style**: Applies Gaussian blur before conversion (for unfinetuned SAM)
**Direct style**: No blur, preserves TransUNet's output (for Phase 2-finetuned SAM)

---

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

All ROI operations are fully differentiable via `grid_sample`.

---

## Quick Start Commands

```bash
# 1. Phase 1: Train TransUNet
python scripts/train_transunet.py \
    --data_root ./dataset/processed \
    --dataset BUSI --fold 0 \
    --vit_pretrained ./pretrained/R50+ViT-B_16.npz

# 2. Phase 2a: Generate augmented data
python scripts/generate_augmented_data.py \
    --data_root ./dataset/processed \
    --output_dir ./dataset/augmented_soft \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --target_samples 100000 --soft_masks --use_sdf

# 3. Phase 2b: Finetune SAM
python scripts/finetune_sam_augmented.py \
    --data_root ./dataset/augmented_soft \
    --dataset COMBINED \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --mask_prompt_style direct --transunet_img_size 224

# 4. Phase 3: End-to-end training (RECOMMENDED with protection)
python scripts/train_e2e.py \
    --data_root ./dataset/processed \
    --datasets BUSI --fold 0 \
    --transunet_checkpoint ./checkpoints/transunet/BUSI/fold_0/best.pth \
    --sam_checkpoint ./checkpoints/sam_finetuned/COMBINED/best_sam.pth \
    --mask_prompt_style direct \
    --transunet_lr 1e-6 --sam_lr 1e-5 \
    --coarse_loss_weight 0.7 --refined_loss_weight 0.3 \
    --transunet_grad_scale 0.1 --transunet_weight_reg 0.01
```

---

## Project Structure

```
UltraRefiner/
├── models/
│   ├── sam_refiner.py        # Differentiable SAM Refiner + ROI Cropper
│   ├── ultra_refiner.py      # End-to-end model
│   └── transunet/            # TransUNet backbone
├── scripts/
│   ├── train_transunet.py    # Phase 1
│   ├── generate_augmented_data.py  # Phase 2a
│   ├── finetune_sam_augmented.py   # Phase 2b
│   └── train_e2e.py          # Phase 3
├── data/
│   ├── dataset.py            # K-fold data loaders
│   └── augmented_dataset.py  # Augmented data loader
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
