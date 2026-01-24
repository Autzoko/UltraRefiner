# UltraRefiner: End-to-End Differentiable Segmentation Refinement

## Abstract

Medical image segmentation, particularly for breast lesions in ultrasound, remains challenging due to low contrast, speckle noise, and ambiguous boundaries. While deep learning methods like TransUNet achieve reasonable performance, they often produce masks with imprecise boundaries, topological errors, or complete failures on difficult cases. The Segment Anything Model (SAM) offers powerful refinement capabilities but requires carefully designed prompts and cannot be trained end-to-end with upstream segmentation networks.

**UltraRefiner** addresses these limitations by introducing a **fully differentiable pipeline** that connects a coarse segmentation network (TransUNet) with SAM for mask refinement. The key innovation is that **gradients flow from SAM's output back through the prompt generation to TransUNet**, enabling joint optimization of both networks. This is achieved through three differentiable prompt extraction mechanisms:

1. **Point Prompts via Soft-Argmax**: The foreground centroid is computed as a probability-weighted average of coordinates, and a background point is extracted from the inverse mask within the bounding box region.

2. **Box Prompts via Weighted Statistics**: Bounding boxes are computed using the mask-weighted mean and standard deviation of coordinates (center ± 2.5σ), ensuring the box tightly covers the predicted region.

3. **Mask Prompts via Direct/Gaussian Conversion**: The soft probability mask is converted to logit space and optionally smoothed with Gaussian blur to match SAM's expected input distribution.

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                    UltraRefiner Pipeline (with ROI Cropping)                   │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   Input Image (224×224)                                                        │
│         │                                                                      │
│         ▼                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐     │
│   │                         TransUNet                                    │     │
│   │     ResNet50 ──► ViT-B/16 Transformer ──► CNN Decoder + Skip        │     │
│   └─────────────────────────────┬───────────────────────────────────────┘     │
│                                 │                                              │
│                                 ▼                                              │
│                    Coarse Mask (Soft Probability)                              │
│                          P(lesion) ∈ [0, 1]                                    │
│                                 │                                              │
│              ┌──────────────────┼──────────────────┐                          │
│              │                  │                  │                           │
│              ▼                  ▼                  ▼                           │
│        ┌──────────┐      ┌───────────┐      ┌──────────┐                      │
│        │  Points  │      │    Box    │      │   Mask   │                      │
│        │ soft-    │      │ weighted  │      │  direct  │                      │
│        │ argmax   │      │ mean±std  │      │ (logits) │                      │
│        └────┬─────┘      └─────┬─────┘      └────┬─────┘                      │
│             │                  │                  │                            │
│             └──────────────────┼──────────────────┘                           │
│                                │                                               │
│   ┌────────────────────────────┼────────────────────────────────────────┐     │
│   │              Differentiable ROI Cropper (Default)                    │     │
│   ├──────────────────────────────────────────────────────────────────────┤     │
│   │                                                                      │     │
│   │   1. Compute soft bounding box from mask (center ± 2.5σ)            │     │
│   │   2. Expand box by 20% (roi_expand_ratio=0.2)                       │     │
│   │   3. Crop image & mask via grid_sample (differentiable)             │     │
│   │   4. Resize ROI to 1024×1024 (full SAM resolution)                  │     │
│   │                                                                      │     │
│   │         ┌─────────────────────────────────────┐                     │     │
│   │         │   ROI Region at 1024×1024           │                     │     │
│   │         │   ┌───────────────────────────┐     │                     │     │
│   │         │   │  Lesion at full resolution │     │                     │     │
│   │         │   │  (higher detail for SAM)   │     │                     │     │
│   │         │   └───────────────────────────┘     │                     │     │
│   │         └─────────────────────────────────────┘                     │     │
│   │                                                                      │     │
│   └──────────────────────────────┬───────────────────────────────────────┘     │
│                                  │                                             │
│                                  ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐     │
│   │                        SAM Refiner                                   │     │
│   │     Image Encoder ──► Prompt Encoder ──► Mask Decoder               │     │
│   │       (frozen)         (trainable)        (trainable)               │     │
│   └─────────────────────────────┬───────────────────────────────────────┘     │
│                                 │                                              │
│                                 ▼                                              │
│                     Refined Mask (ROI space)                                   │
│                                 │                                              │
│   ┌─────────────────────────────┼───────────────────────────────────────┐     │
│   │              Differentiable Paste Back                               │     │
│   │   grid_sample inverse: ROI mask → Original image space              │     │
│   └─────────────────────────────┬───────────────────────────────────────┘     │
│                                 │                                              │
│                                 ▼                                              │
│                         Final Refined Mask                                     │
│                                                                                │
│   ◄─── Gradient Flow: L_refined → Paste → SAM → Crop → Prompts → TransUNet    │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Why ROI Cropping is Default

ROI cropping provides several advantages:
1. **Higher effective resolution**: The lesion region is processed at full 1024×1024 SAM resolution
2. **Better boundary refinement**: SAM sees more detail in the lesion area
3. **Fully differentiable**: Both crop and paste-back use `grid_sample` for gradient flow
4. **Consistent training**: Both Phase 2 and Phase 3 use the same ROI pipeline

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
    --use_roi_crop \
    --roi_expand_ratio 0.2 \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4
```

**Key flags for Phase 3 alignment:**

| Flag | Value | Purpose |
|------|-------|---------|
| `--mask_prompt_style` | `direct` | Since soft masks already have smooth boundaries, no extra blur needed. **Must match Phase 3 setting.** |
| `--transunet_img_size` | `224` | **CRITICAL**: Simulates Phase 3 resolution path. Coarse masks are first resized to 224×224, then upscaled to 1024×1024. This bilinear interpolation creates additional smoothing that Phase 3 naturally has. |
| `--use_roi_crop` | flag | **DEFAULT**: Enables ROI cropping mode. Crops lesion region and processes at full 1024×1024 resolution. |
| `--roi_expand_ratio` | `0.2` | Expands bounding box by 20% to include context around lesion. |

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

#### Recommended Command (with ROI + TransUNet Protection)

During E2E training with ROI mode, gradients from SAM create stronger coupling and can destabilize TransUNet. Use these protection mechanisms:

```bash
# RECOMMENDED: Full E2E training with ROI + TransUNet protection
python scripts/train_e2e.py \
    --data_root ./dataset/processed \
    --datasets BUSI \
    --fold 0 \
    --transunet_checkpoint ./checkpoints/transunet/BUSI/fold_0/best.pth \
    --sam_checkpoint ./checkpoints/sam_finetuned/COMBINED/best_sam.pth \
    --mask_prompt_style direct \
    --use_roi_crop \
    --roi_expand_ratio 0.2 \
    --max_epochs 100 \
    --transunet_lr 1e-6 \
    --sam_lr 1e-5 \
    --coarse_loss_weight 0.8 \
    --refined_loss_weight 0.2 \
    --transunet_grad_scale 0.01 \
    --transunet_weight_reg 0.1 \
    --grad_clip 1.0
```

#### TransUNet Protection Options (Stronger for ROI Mode)

ROI mode creates stronger gradient coupling because the crop coordinates depend on the mask. Use these stronger protection values:

| Flag | ROI Mode | Non-ROI Mode | Purpose |
|------|----------|--------------|---------|
| `--transunet_lr` | `1e-6` | `1e-6` | Lower learning rate prevents large weight updates |
| `--coarse_loss_weight` | `0.8` | `0.7` | Higher weight maintains TransUNet quality |
| `--refined_loss_weight` | `0.2` | `0.3` | Lower SAM influence on total loss |
| `--transunet_grad_scale` | `0.01` | `0.1` | Scales gradients from SAM (1% for ROI, 10% for non-ROI) |
| `--transunet_weight_reg` | `0.1` | `0.01` | L2 penalty anchors weights to Phase 1 checkpoint |
| `--freeze_transunet_epochs` | `5-10` | `0` | Optional: let SAM adapt first before joint training |

The script automatically:
1. **Evaluates TransUNet baseline** before training starts
2. **Compares performance** at each epoch (warns if TransUNet degrades)
3. **Logs to TensorBoard** for monitoring

#### Basic Command (ROI without protection - NOT recommended)

```bash
python scripts/train_e2e.py \
    --data_root ./dataset/processed \
    --datasets BUSI \
    --fold 0 \
    --transunet_checkpoint ./checkpoints/transunet/BUSI/fold_0/best.pth \
    --sam_checkpoint ./checkpoints/sam_finetuned/COMBINED/best_sam.pth \
    --mask_prompt_style direct \
    --use_roi_crop \
    --roi_expand_ratio 0.2 \
    --max_epochs 100
```

**Phase 2 → Phase 3 Consistency Requirements:**

| Setting | Phase 2 | Phase 3 | Why |
|---------|---------|---------|-----|
| `mask_prompt_style` | `direct` | `direct` | Same mask preprocessing |
| Resolution path | 224→1024 | 224→1024 | Same interpolation smoothing |
| `use_roi_crop` | `True` | `True` | **DEFAULT**: Same cropping behavior |
| `roi_expand_ratio` | `0.2` | `0.2` | Same crop boundaries |

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
    --use_roi_crop \
    --roi_expand_ratio 0.2 \
    --transunet_grad_scale 0.01 \
    --transunet_weight_reg 0.1 \
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

## ROI Cropping Mode (Default)

ROI cropping is the **default and recommended** mode. It focuses SAM computation on the lesion region at full 1024×1024 resolution.

### How ROI Cropping Works

1. **Compute soft bounding box** from the coarse mask using weighted statistics (center ± 2.5σ)
2. **Expand box by 20%** (`roi_expand_ratio=0.2`) to include surrounding context
3. **Crop image and mask** via `grid_sample` (fully differentiable)
4. **Resize ROI to 1024×1024** for SAM processing at full resolution
5. **SAM refines** the cropped mask
6. **Paste back** the refined mask to original image space via inverse `grid_sample`

### ROI Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--use_roi_crop` | **Recommended** | Enable ROI cropping mode |
| `--roi_expand_ratio` | `0.2` | Expand bounding box by this ratio (20% = include context) |

### Disabling ROI (Not Recommended)

If you want to disable ROI and use full-image processing:

```bash
# Phase 2 without ROI
python scripts/finetune_sam_augmented.py \
    --data_root ./dataset/augmented_soft \
    --dataset COMBINED \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --mask_prompt_style direct --transunet_img_size 224
    # Note: no --use_roi_crop flag

# Phase 3 without ROI (must match Phase 2)
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

# 3. Phase 2b: Finetune SAM (with ROI - default)
python scripts/finetune_sam_augmented.py \
    --data_root ./dataset/augmented_soft \
    --dataset COMBINED \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --mask_prompt_style direct --transunet_img_size 224 \
    --use_roi_crop --roi_expand_ratio 0.2

# 4. Phase 3: End-to-end training (with ROI + protection - RECOMMENDED)
python scripts/train_e2e.py \
    --data_root ./dataset/processed \
    --datasets BUSI --fold 0 \
    --transunet_checkpoint ./checkpoints/transunet/BUSI/fold_0/best.pth \
    --sam_checkpoint ./checkpoints/sam_finetuned/COMBINED/best_sam.pth \
    --mask_prompt_style direct \
    --use_roi_crop --roi_expand_ratio 0.2 \
    --transunet_lr 1e-6 --sam_lr 1e-5 \
    --coarse_loss_weight 0.8 --refined_loss_weight 0.2 \
    --transunet_grad_scale 0.01 --transunet_weight_reg 0.1
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
