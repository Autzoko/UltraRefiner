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

#### Option A: Online Augmentation (RECOMMENDED - No Pre-generation)

The new **online augmentation system** applies mask augmentation on-the-fly during training:

```bash
python scripts/finetune_sam_online.py \
    --data_root ./dataset/processed \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --output_dir ./checkpoints/sam_finetuned \
    --augmentor_preset default \
    --mask_prompt_style direct \
    --transunet_img_size 224 \
    --use_roi_crop \
    --roi_expand_ratio 0.2 \
    --epochs 50 \
    --batch_size 4
```

**Benefits of online augmentation:**
- No disk storage required for augmented masks
- New augmentation every epoch (unlimited diversity)
- 12 primary error types simulating TransUNet failures
- Soft mask conversion matching TransUNet output distribution

**12 Primary Error Types:**

| # | Error Type | Prob | Description |
|---|------------|------|-------------|
| 1 | Identity/Near-Perfect | 15% | Preserve good predictions |
| 2 | Over-Segmentation | 17% | 1.2x-3x area expansion |
| 3 | Giant Over-Segmentation | 10% | 3x-20x area expansion |
| 4 | Under-Segmentation | 17% | 0.4x-0.9x area shrinkage |
| 5 | Missing Chunk | 12% | 5-30% wedge/blob cutout |
| 6 | Internal Holes | 10% | 1-3 holes (2-20% each) |
| 7 | Bridge/Adhesion | 10% | Thin band + FP blob |
| 8 | False Positive Islands | 17% | 1-30 scattered blobs |
| 9 | Fragmentation | 10% | 1-5 cuts through mask |
| 10 | Shift/Wrong Location | 10% | 5-30% translation |
| 11 | Empty Prediction | 4% | Complete miss |
| 12 | Noise-Only Scatter | 3% | Pure FP noise |

**Augmentor Presets:**

| Preset | Description |
|--------|-------------|
| `default` | Balanced distribution (recommended) |
| `mild` | More identity (30%), fewer severe errors |
| `severe` | More extreme failures (giant, empty, scatter) |
| `boundary_focus` | 50% over/under-segmentation |
| `structural` | 40% holes + missing + fragmentation |

#### Option B: Hybrid Training (Real Predictions + Augmented GT)

This option combines **real TransUNet predictions** with **augmented GT masks** for the best of both worlds:
- Real predictions capture actual TransUNet failure modes
- Augmented GT provides additional diversity and edge cases

**Step 1: Generate TransUNet Predictions**
```bash
# Generate predictions on all training data
python scripts/generate_transunet_predictions.py \
    --data_root ./dataset/processed \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --checkpoint_dir ./checkpoints/transunet \
    --output_dir ./dataset/transunet_preds \
    --use_all_folds  # Ensemble all 5 folds for better predictions
```

This creates:
```
./dataset/transunet_preds/
└── {dataset}/
    └── train/
        ├── images/       (symlinks to original)
        ├── masks/        (symlinks to GT)
        └── coarse_masks/ (TransUNet soft predictions as .npy)
```

**Step 2: Train with Hybrid Data**
```bash
python scripts/finetune_sam_hybrid.py \
    --gt_data_root ./dataset/processed \
    --pred_data_root ./dataset/transunet_preds \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --real_ratio 0.5 \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --output_dir ./checkpoints/sam_finetuned_hybrid \
    --augmentor_preset default \
    --mask_prompt_style direct --transunet_img_size 224 \
    --use_roi_crop --roi_expand_ratio 0.5 \
    --use_amp --fast_soft_mask \
    --epochs 300 --batch_size 8
```

**Key parameter: `--real_ratio`**
| Value | Meaning |
|-------|---------|
| `0.5` | 50% real predictions, 50% augmented GT (balanced) |
| `0.7` | 70% real, 30% augmented (favor real failures) |
| `0.3` | 30% real, 70% augmented (favor diversity) |

#### Option C: Pre-generated Augmented Data

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

## Gated Residual Refinement (Alternative to Phase 2)

If you want to skip Phase 2 but still get meaningful improvements, use **Gated Residual Refinement**. This constrains SAM to act as a controlled error corrector instead of directly replacing the coarse prediction.

### How It Works

```
Standard:  final = SAM(coarse)                    # Direct replacement
Gated:     final = coarse + gate × (SAM - coarse) # Controlled correction
```

The gate is computed from **coarse mask uncertainty**:
- When coarse ≈ 0.5 (uncertain): gate ≈ 1 → trust SAM's correction
- When coarse ≈ 0 or 1 (confident): gate ≈ 0 → preserve coarse prediction

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Gated Residual Refinement                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   coarse_mask (from TransUNet)                                       │
│        │                                                             │
│        ├───────────────────────┐                                     │
│        │                       │                                     │
│        ▼                       ▼                                     │
│   ┌─────────┐           ┌──────────────┐                            │
│   │   SAM   │           │ Uncertainty  │                            │
│   │ Refiner │           │    Gate      │                            │
│   └────┬────┘           │ 1-|2p-1|^γ   │                            │
│        │                └──────┬───────┘                            │
│        ▼                       │                                     │
│   sam_output                   │                                     │
│        │                       │                                     │
│        └───────────┬───────────┘                                     │
│                    │                                                 │
│                    ▼                                                 │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  final = coarse + gate × (sam_output - coarse)              │   │
│   │                                                              │   │
│   │  gate ≈ 0 (confident): final ≈ coarse (preserve)            │   │
│   │  gate ≈ 1 (uncertain): final ≈ sam    (correct)             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Command-Line Usage

Enable gated refinement by adding `--use_gated_refinement` flag:

```bash
# Skip Phase 2 with gated refinement (RECOMMENDED for unfinetuned SAM)
python scripts/train_e2e.py \
    --data_root ./dataset/processed \
    --datasets BUSI \
    --fold 0 \
    --transunet_checkpoint ./checkpoints/transunet/BUSI/fold_0/best.pth \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --mask_prompt_style gaussian \
    --use_roi_crop \
    --roi_expand_ratio 0.2 \
    --use_gated_refinement \
    --gate_type uncertainty \
    --gate_gamma 1.0 \
    --gate_max 0.5 \
    --max_epochs 100 \
    --coarse_loss_weight 0.8 \
    --refined_loss_weight 0.2
```

### Command-Line Arguments for Gated Refinement

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_gated_refinement` | `False` | Enable gated residual refinement |
| `--gate_type` | `uncertainty` | Gate type: `uncertainty`, `learned`, `hybrid` |
| `--gate_gamma` | `1.0` | Uncertainty curve shape (higher = more aggressive) |
| `--gate_min` | `0.0` | Minimum gate value (0 = fully preserve confident) |
| `--gate_max` | `0.8` | Maximum gate value (caps correction strength) |

### Python API Usage

```python
from models.ultra_refiner import build_gated_ultra_refiner

model = build_gated_ultra_refiner(
    vit_name='R50-ViT-B_16',
    img_size=224,
    num_classes=2,
    sam_model_type='vit_b',
    transunet_checkpoint='./checkpoints/transunet/BUSI/fold_0/best.pth',
    sam_checkpoint='./pretrained/medsam_vit_b.pth',
    gate_type='uncertainty',
    gate_gamma=1.0,
    gate_min=0.0,
    gate_max=0.5,
    mask_prompt_style='gaussian',
    use_roi_crop=True,
    roi_expand_ratio=0.2,
)
```

### Gate Types

| Type | Description | Parameters |
|------|-------------|------------|
| `uncertainty` | Based on coarse mask confidence: `gate = 1 - |2*coarse - 1|^γ` | No learnable params |
| `learned` | CNN that predicts where corrections should be applied | ~3K learnable params |
| `hybrid` | `uncertainty × learned` - combines both approaches | ~3K learnable params |

### Gate Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `gate_gamma` | `1.0` | Higher = more aggressive (only very uncertain regions) |
| `gate_min` | `0.0` | Minimum correction (0 = fully preserve confident regions) |
| `gate_max` | `0.8` | Maximum correction (0.8 = cap at 80%) |

### When to Use Gated Refinement

| Scenario | Recommendation |
|----------|----------------|
| Skip Phase 2 completely | Use gated with `gate_max=0.5` (conservative) |
| Unfinetuned SAM makes wild guesses | Use gated with `gate_max=0.3-0.5` |
| Phase 2-finetuned SAM | Standard refinement (no gating needed) |
| TransUNet already very good | Use gated with `gate_max=0.3` to preserve |

### Output Format

```python
output = model(image)

# Standard outputs
coarse_mask = output['coarse_mask']      # TransUNet prediction (B, H, W)
refined_mask = output['refined_mask']    # Final gated output (B, H, W) ← use this

# Gated refinement outputs
gate = output['gate']                    # Where corrections applied (B, H, W)
residual = output['residual']            # SAM's proposed changes (B, H, W)
sam_mask = output['sam_mask']            # Raw SAM output before gating (B, H, W)
```

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

# 2. Phase 2: Finetune SAM (CHOOSE ONE OPTION)

# Option A: Online Augmentation (RECOMMENDED - no pre-generation needed)
python scripts/finetune_sam_online.py \
    --data_root ./dataset/processed \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --output_dir ./checkpoints/sam_finetuned \
    --augmentor_preset default \
    --mask_prompt_style direct --transunet_img_size 224 \
    --use_roi_crop --roi_expand_ratio 0.2 \
    --epochs 50 --batch_size 4

# Option B: Hybrid Training (real TransUNet predictions + augmented GT)
python scripts/generate_transunet_predictions.py \
    --data_root ./dataset/processed \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --checkpoint_dir ./checkpoints/transunet \
    --output_dir ./dataset/transunet_preds \
    --use_all_folds

python scripts/finetune_sam_hybrid.py \
    --gt_data_root ./dataset/processed \
    --pred_data_root ./dataset/transunet_preds \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --real_ratio 0.5 \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --mask_prompt_style direct --transunet_img_size 224 \
    --use_roi_crop --roi_expand_ratio 0.5 \
    --use_amp --fast_soft_mask

# Option C: Pre-generated Data (requires generate_augmented_data.py first)
python scripts/generate_augmented_data.py \
    --data_root ./dataset/processed \
    --output_dir ./dataset/augmented_soft \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --target_samples 100000 --soft_masks --use_sdf

python scripts/finetune_sam_augmented.py \
    --data_root ./dataset/augmented_soft \
    --dataset COMBINED \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --mask_prompt_style direct --transunet_img_size 224 \
    --use_roi_crop --roi_expand_ratio 0.2

# 3. Phase 3: End-to-end training (with ROI + protection - RECOMMENDED)
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

# Alternative: Skip Phase 2 with Gated Refinement
python scripts/train_e2e.py \
    --data_root ./dataset/processed \
    --datasets BUSI --fold 0 \
    --transunet_checkpoint ./checkpoints/transunet/BUSI/fold_0/best.pth \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --mask_prompt_style gaussian \
    --use_roi_crop --roi_expand_ratio 0.2 \
    --use_gated_refinement \
    --gate_type uncertainty --gate_max 0.5 \
    --coarse_loss_weight 0.8 --refined_loss_weight 0.2 \
    --freeze_transunet_epochs 10
```

---

## Mask Augmentation Module

The mask augmentation module (`data/mask_augmentation.py`) provides comprehensive simulation of TransUNet failure modes for Phase 2 training.

### Python API

```python
from data import MaskAugmentor, create_augmentor, AUGMENTOR_PRESETS

# Create augmentor with preset
augmentor = create_augmentor(
    preset='default',           # 'default', 'mild', 'severe', 'boundary_focus', 'structural'
    soft_mask_prob=0.8,         # Probability of soft mask conversion
    soft_mask_temperature=(2.0, 8.0),  # Temperature range for soft masks
    secondary_prob=0.5,         # Probability of secondary perturbations
)

# Apply augmentation to GT mask
gt_mask = np.array(Image.open('mask.png').convert('L')) / 255.0
coarse_mask, aug_info = augmentor(gt_mask)

# aug_info contains:
print(aug_info['error_type'])    # e.g., 'over_segmentation'
print(aug_info['secondary'])     # e.g., ['boundary_jitter']
print(aug_info['soft'])          # True if soft mask applied
print(aug_info['dice'])          # Dice score between coarse and GT

# Force specific error type for testing
coarse_mask, _ = augmentor(gt_mask, force_error_type='internal_holes')
```

### Online Augmented Dataset

```python
from data import OnlineAugmentedDataset, get_online_augmented_dataloaders

# Single dataset
dataset = OnlineAugmentedDataset(
    data_root='./dataset/processed',
    dataset_name='BUSI',
    img_size=1024,              # SAM input size
    transunet_img_size=224,     # Resolution path matching
    augmentor_preset='default',
    soft_mask_prob=0.8,
    split_ratio=0.9,            # Train/val split
    is_train=True,
)

# Combined dataloaders (multiple datasets)
train_loader, val_loader = get_online_augmented_dataloaders(
    data_root='./dataset/processed',
    dataset_names=['BUSI', 'BUSBRA', 'BUS'],
    batch_size=4,
    img_size=1024,
    transunet_img_size=224,
    augmentor_preset='default',
    num_workers=4,
)

# Each batch:
for batch in train_loader:
    image = batch['image']          # (B, 3, 1024, 1024) - SAM normalized
    label = batch['label']          # (B, 1024, 1024) - GT mask
    coarse = batch['coarse_mask']   # (B, 1024, 1024) - Augmented mask
    dice = batch['dice']            # Dice(coarse, GT)
    error_type = batch['error_type'] # Primary error applied
```

### Hybrid Dataset API

```python
from data import HybridDataset, get_hybrid_dataloaders

# Create hybrid dataloaders
train_loader, val_loader = get_hybrid_dataloaders(
    gt_data_root='./dataset/processed',
    pred_data_root='./dataset/transunet_preds',
    dataset_names=['BUSI', 'BUSBRA', 'BUS'],
    batch_size=4,
    img_size=1024,
    transunet_img_size=224,
    real_ratio=0.5,          # 50% real predictions, 50% augmented
    augmentor_preset='default',
    use_fast_soft_mask=True,
    num_workers=8,
)

# Each batch:
for batch in train_loader:
    image = batch['image']          # (B, 3, 1024, 1024)
    label = batch['label']          # (B, 1024, 1024)
    coarse = batch['coarse_mask']   # (B, 1024, 1024)
    source = batch['source']        # 'real' or 'augmented'
    dice = batch['dice']            # Dice(coarse, GT)
    error_type = batch['error_type']
```

### Soft Mask Conversion

The soft mask conversion uses signed distance transform to create realistic probability maps:

```
signed_dist = distance_inside - distance_outside
P(x) = sigmoid(signed_dist(x) / temperature)
```

- **Low temperature (2.0)**: Sharp boundaries, high confidence
- **High temperature (8.0)**: Fuzzy boundaries, gradual transitions

This matches TransUNet's soft probability output distribution.

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
│   ├── generate_augmented_data.py  # Phase 2a (pre-generation)
│   ├── generate_transunet_predictions.py  # Generate TransUNet preds for hybrid [NEW]
│   ├── finetune_sam_augmented.py   # Phase 2b (pre-generated data)
│   ├── finetune_sam_online.py      # Phase 2b (online augmentation) [NEW]
│   ├── finetune_sam_hybrid.py      # Phase 2b (hybrid: real + augmented) [NEW]
│   └── train_e2e.py          # Phase 3
├── data/
│   ├── dataset.py            # K-fold data loaders
│   ├── augmented_dataset.py  # Augmented data loader (pre-generated)
│   ├── mask_augmentation.py  # Mask augmentor with 12 error types [NEW]
│   ├── online_augmented_dataset.py  # Online augmentation dataset [NEW]
│   └── hybrid_dataset.py     # Hybrid dataset (real + augmented) [NEW]
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
