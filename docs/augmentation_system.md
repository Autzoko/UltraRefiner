# Data Augmentation System for SAM Refiner Training

## Overview

This document describes the data augmentation system designed to expand ~3,000 training samples to ~100,000 samples with controlled Dice score distribution. The augmentation methods simulate realistic segmentation model failures to train the SAM Refiner's ability to:
1. **Correct** low-quality segmentations
2. **Preserve** high-quality segmentations (know when NOT to modify)

## Key Design Principles

1. **Simulate realistic failure modes** from actual segmentation models
2. **Condition augmentation intensity** on lesion SIZE and SHAPE COMPLEXITY
3. **Include PERFECT masks** so the Refiner learns "when not to modify"
4. **Mix perfect, mildly corrupted, and heavily corrupted masks**
5. **Quality-aware loss** with change penalty weighted by input quality

## Target Distribution

| Dice Range | Percentage | Count (~100K total) | Description |
|------------|------------|---------------------|-------------|
| 1.0 (Perfect) | 10%     | ~10,000             | Unchanged GT - preservation learning |
| 0.90-1.00  | 15%        | ~15,000             | Minor artifacts - mostly preserve |
| 0.80-0.90  | 40%        | ~40,000             | Moderate errors - needs refinement |
| 0.60-0.80  | 35%        | ~35,000             | Severe failures - strong correction |

## Quality-Aware Training

### The "When Not to Modify" Problem

A critical challenge in training a refinement model is ensuring it learns:
- **When TO modify**: Low-quality inputs need strong correction
- **When NOT to modify**: High-quality inputs should be preserved

### Quality-Aware Loss Function

The loss function includes a **change penalty** weighted by input quality:

```
L_total = L_seg + λ * L_change_penalty

where:
  L_seg = L_bce + L_dice + 0.5 * L_focal
  L_change_penalty = ||refined - coarse||² * quality_weight(Dice(coarse, GT))

  quality_weight(d) = clamp((d - 0.6) / 0.4, 0, 1)
    - Dice = 0.6 → weight = 0.0 (no penalty for changes)
    - Dice = 1.0 → weight = 1.0 (strong penalty for changes)
```

### Effect of Change Penalty

| Input Quality | Change Penalty | Behavior |
|--------------|----------------|----------|
| Dice = 1.0 (perfect) | Maximum | Preserve input, minimal refinement |
| Dice = 0.9 | High | Light refinement only |
| Dice = 0.8 | Medium | Moderate refinement allowed |
| Dice = 0.6 | Zero | Strong refinement encouraged |

## Segmentation Failure Analysis

### Common Failure Patterns in Medical Image Segmentation

Based on analysis of real segmentation model outputs, we identified the following failure categories:

### Under-segmentation Failures

#### 1. Boundary Erosion
- **Cause**: Conservative predictions, feature map resolution loss
- **Appearance**: Object boundaries shrunk inward
- **Dice impact**: 5-40% reduction

#### 2. Partial Breakage (NEW)
- **Cause**: Thin connection loss, attention gaps
- **Appearance**: Object split into disconnected parts
- **Dice impact**: 10-30% reduction
- **Size-conditioned**: More likely for irregular shapes

#### 3. Small Lesion Disappearance (NEW - CRITICAL)
- **Cause**: Pooling operations, class imbalance
- **Appearance**: Tiny lesions completely missed or severely shrunk
- **Dice impact**: 50-100% reduction
- **Size-conditioned**: Only applies to tiny lesions (<500 pixels)

#### 4. Partial Dropout
- **Cause**: Attention failures, feature dropout
- **Appearance**: Missing large regions
- **Dice impact**: 10-40% reduction

### Over-segmentation Failures

#### 5. Boundary Dilation
- **Cause**: Over-confident predictions, upsampling artifacts
- **Appearance**: Boundaries expanded into background
- **Dice impact**: 5-35% reduction

#### 6. Attachment to Nearby Structures (NEW)
- **Cause**: Similar texture confusion, upsampling errors
- **Appearance**: Unintended connection to adjacent anatomy
- **Dice impact**: 5-25% reduction

#### 7. Artificial Bridges (NEW)
- **Cause**: Interpolation artifacts, feature leakage
- **Appearance**: Thin artificial connections to nearby regions
- **Dice impact**: 3-15% reduction

### Boundary Artifact Failures

#### 8. Elastic Deformation
- **Cause**: Checkerboard artifacts, spatial inconsistency
- **Appearance**: Wavy, distorted boundaries
- **Dice impact**: 5-25% reduction

#### 9. Contour Jitter (NEW)
- **Cause**: Aliasing, quantization at boundaries
- **Appearance**: Pixel-level noise along edge band
- **Dice impact**: 2-10% reduction

#### 10. Edge Roughening
- **Cause**: Low resolution, pixelation
- **Appearance**: Jagged, staircase boundaries
- **Dice impact**: 2-10% reduction

### Internal Failures

#### 11. Internal Holes (Low-contrast)
- **Cause**: Central regions confused with background
- **Appearance**: Missing internal regions
- **Dice impact**: 5-30% reduction
- **Size-conditioned**: More likely for large lesions

### False Positive Failures

#### 12. False Positive Islands (NEW)
- **Cause**: Speckle noise, satellite lesion confusion
- **Appearance**: Small spurious blobs NEAR the lesion
- **Dice impact**: 3-15% reduction

## Size and Shape Conditioning

Augmentation selection is **conditioned on lesion properties**:

| Lesion Property | Augmentations More Likely | Rationale |
|----------------|---------------------------|-----------|
| Tiny (<500 px) | Small lesion disappearance, extreme shrinkage | Most commonly missed by models |
| Small (<2000 px) | Shrinkage, partial breakage | Feature resolution loss |
| Irregular (circularity <0.5) | Partial breakage, boundary artifacts | Complex shapes hard to segment |
| Large | Internal holes, partial dropout | More area for internal failures |

### Lesion Analysis Metrics

```python
def analyze_lesion(mask):
    returns {
        'area': pixel count,
        'diameter': equivalent circular diameter,
        'circularity': 4π × area / perimeter² (1.0 = perfect circle),
        'is_tiny': area < 500,
        'is_small': area < 2000,
        'is_irregular': circularity < 0.5,
        'complexity': overall shape complexity (0-1)
    }
```

## Flow Chart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DATA AUGMENTATION PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  INPUT: Original Dataset (~3,000 samples)                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                          │
│  │   Image     │  │  GT Mask    │  │  Metadata   │                          │
│  │  (H x W)    │  │  (H x W)    │  │   (name)    │                          │
│  └─────────────┘  └─────────────┘  └─────────────┘                          │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Calculate Augmentations Per Sample                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  target_samples = 100,000                                              │ │
│  │  original_samples = 3,000                                              │ │
│  │  augs_per_sample = 100,000 / 3,000 ≈ 33                               │ │
│  │                                                                        │ │
│  │  Per-sample distribution:                                              │ │
│  │    - Low Dice (0.6-0.8):   33 × 30% ≈ 10 augmentations                │ │
│  │    - Medium Dice (0.8-0.9): 33 × 50% ≈ 17 augmentations               │ │
│  │    - High Dice (0.9+):     33 × 20% ≈  6 augmentations                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: For Each Original Sample                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  FOR sample in original_dataset:                                       │ │
│  │      load image, gt_mask                                               │ │
│  │                                                                        │ │
│  │      # Generate Low Dice augmentations (severe failures)               │ │
│  │      FOR i in range(10):                                               │ │
│  │          coarse_mask = generate_augmentation(gt_mask,                  │ │
│  │                                              target_dice=(0.6, 0.8))   │ │
│  │          save(image, gt_mask, coarse_mask)                             │ │
│  │                                                                        │ │
│  │      # Generate Medium Dice augmentations (moderate errors)            │ │
│  │      FOR i in range(17):                                               │ │
│  │          coarse_mask = generate_augmentation(gt_mask,                  │ │
│  │                                              target_dice=(0.8, 0.9))   │ │
│  │          save(image, gt_mask, coarse_mask)                             │ │
│  │                                                                        │ │
│  │      # Generate High Dice augmentations (minor artifacts)              │ │
│  │      FOR i in range(6):                                                │ │
│  │          coarse_mask = generate_augmentation(gt_mask,                  │ │
│  │                                              target_dice=(0.9, 1.0))   │ │
│  │          save(image, gt_mask, coarse_mask)                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Augmentation Selection by Dice Target                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────┐ │ │
│  │  │ LOW DICE (0.6-0.8)  │  │MEDIUM DICE (0.8-0.9)│  │HIGH DICE (0.9+) │ │ │
│  │  ├─────────────────────┤  ├─────────────────────┤  ├─────────────────┤ │ │
│  │  │ Heavy augmentation: │  │Moderate augmentation│  │Light augmentat. │ │ │
│  │  │ - Large erosion     │  │ - Medium erosion    │  │ - Small erosion │ │ │
│  │  │ - Large dilation    │  │ - Medium dilation   │  │ - Light noise   │ │ │
│  │  │ - Multiple holes    │  │ - Few small holes   │  │ - Edge jitter   │ │ │
│  │  │ - Large false pos   │  │ - Small false pos   │  │ - Slight blur   │ │ │
│  │  │ - Partial dropout   │  │ - Elastic deform    │  │                 │ │ │
│  │  │ - 2-4 combined      │  │ - 1-3 combined      │  │ - 1-2 combined  │ │ │
│  │  └─────────────────────┘  └─────────────────────┘  └─────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Augmentation Generation (compose_augmentations)                     │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  def compose_augmentations(gt_mask, target_dice_range):                │ │
│  │                                                                        │ │
│  │      FOR attempt in range(max_attempts=15):                            │ │
│  │          result = gt_mask.copy()                                       │ │
│  │                                                                        │ │
│  │          # Select augmentations based on target range                  │ │
│  │          augmentation_pool = get_pool_for_target(target_dice_range)    │ │
│  │          num_augs = sample_num_augmentations(target_dice_range)        │ │
│  │                                                                        │ │
│  │          # Apply selected augmentations                                │ │
│  │          FOR aug_name, base_intensity in selected_augs:                │ │
│  │              intensity = base_intensity × random(0.7, 1.3)             │ │
│  │              result = apply_augmentation(result, aug_name, intensity)  │ │
│  │                                                                        │ │
│  │          # Check if achieved target                                    │ │
│  │          dice = compute_dice(result, gt_mask)                          │ │
│  │          IF target_min ≤ dice ≤ target_max:                            │ │
│  │              RETURN result, dice, applied_augmentations                │ │
│  │                                                                        │ │
│  │      RETURN best_result, best_dice, best_augmentations                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: Augmented Dataset (~100,000 samples)                                │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  augmented_data/                                                       │ │
│  │  └── {dataset}/                                                        │ │
│  │      ├── images/           # Original images (copied)                  │ │
│  │      │   └── {name}_aug{idx}_{level}.png                              │ │
│  │      ├── masks/            # Ground truth masks (copied)               │ │
│  │      │   └── {name}_aug{idx}_{level}.png                              │ │
│  │      ├── coarse_masks/     # Augmented coarse masks                    │ │
│  │      │   └── {name}_aug{idx}_{level}.png                              │ │
│  │      └── metadata.json     # Sample information and statistics         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Augmentation Method Details

### Method 1: Boundary Erosion

```
Input:  GT mask (binary)
Output: Eroded mask simulating under-segmentation

Algorithm:
1. Calculate mask diameter: d = 2 × √(area / π)
2. Erosion amount: pixels = d × (0.02 + 0.13 × intensity)
3. Create elliptical kernel of size (2×pixels + 1)
4. Apply morphological erosion
5. Add boundary noise for realism

Intensity mapping:
- intensity=0.1 → ~2% boundary loss → Dice ~0.95
- intensity=0.5 → ~8% boundary loss → Dice ~0.80
- intensity=1.0 → ~15% boundary loss → Dice ~0.60
```

### Method 2: Boundary Dilation

```
Input:  GT mask (binary)
Output: Dilated mask simulating over-segmentation

Algorithm:
1. Calculate mask diameter: d = 2 × √(area / π)
2. Dilation amount: pixels = d × (0.02 + 0.13 × intensity)
3. Create elliptical kernel of size (2×pixels + 1)
4. Apply morphological dilation
5. Apply Gaussian blur for soft boundaries

Intensity mapping:
- intensity=0.1 → ~2% boundary expansion → Dice ~0.95
- intensity=0.5 → ~8% boundary expansion → Dice ~0.82
- intensity=1.0 → ~15% boundary expansion → Dice ~0.65
```

### Method 3: Elastic Deformation

```
Input:  GT mask (H × W)
Output: Elastically deformed mask

Algorithm:
1. Generate random displacement fields: dx, dy ~ N(0, α²)
   where α = 5 + 25 × intensity
2. Smooth displacement with Gaussian: σ = 3 + 7 × intensity
3. Create coordinate grid: x, y = meshgrid(W, H)
4. Apply displacement: new_x = x + dx, new_y = y + dy
5. Remap mask using bilinear interpolation

Intensity mapping:
- intensity=0.2 → subtle wobble → Dice ~0.92
- intensity=0.5 → noticeable distortion → Dice ~0.85
- intensity=0.8 → severe distortion → Dice ~0.70
```

### Method 4: Add Holes (False Negatives)

```
Input:  GT mask
Output: Mask with internal holes

Algorithm:
1. Find foreground coordinates
2. num_holes = 1 + 5 × intensity
3. hole_area = mask_area × (0.03 + 0.12 × intensity) / num_holes
4. FOR each hole:
   a. Select random center inside foreground
   b. hole_radius = √(hole_area / π) ± random variation
   c. Create elliptical hole mask
   d. Apply hole: result[hole] *= random(0, 0.2)

Intensity mapping:
- intensity=0.2 → 1-2 small holes → Dice ~0.93
- intensity=0.5 → 2-3 medium holes → Dice ~0.82
- intensity=0.8 → 4-5 large holes → Dice ~0.68
```

### Method 5: False Positive Blobs

```
Input:  GT mask
Output: Mask with spurious detections

Algorithm:
1. Find background coordinates
2. num_blobs = 1 + 4 × intensity
3. blob_area = mask_area × (0.01 + 0.07 × intensity) / num_blobs
4. FOR each blob:
   a. Select random center in background
   b. blob_radius = √(blob_area / π) ± random variation
   c. Create elliptical blob mask
   d. Add blob: result[blob] = max(result[blob], random(0.6, 1.0))

Intensity mapping:
- intensity=0.2 → 1 small blob → Dice ~0.95
- intensity=0.5 → 2-3 medium blobs → Dice ~0.85
- intensity=0.8 → 4-5 large blobs → Dice ~0.72
```

### Method 6: Partial Dropout

```
Input:  GT mask
Output: Mask with missing regions

Algorithm:
1. dropout_fraction = 0.05 + 0.30 × intensity
2. Select strategy: {corner, side, random_region}
3. IF strategy == corner:
   - Find object centroid
   - Select random quadrant
   - Apply dropout to quadrant intersection with object
4. ELIF strategy == side:
   - Find object bounding box
   - Select random side (top/bottom/left/right)
   - Apply dropout to side fraction
5. ELSE (random_region):
   - Select random center inside object
   - dropout_radius = √(area × dropout_fraction / π)
   - Apply elliptical dropout

Intensity mapping:
- intensity=0.2 → 10% dropout → Dice ~0.88
- intensity=0.5 → 20% dropout → Dice ~0.75
- intensity=0.8 → 30% dropout → Dice ~0.62
```

## SDF-Based Augmentation (Recommended)

The system supports a mathematically grounded SDF-based augmentation backend that operates
in the continuous Signed Distance Function domain rather than directly on binary masks.

### Mathematical Framework

```
GT mask M → SDF φ (zero level set defines contour)
Perturbation: φ'(x,y) = φ(x,y) + δ(x,y)

where δ(x,y) = c + G(x,y)
  c = global offset (uniform erosion/dilation)
  G(x,y) = Gaussian Random Field (spatially varying boundary shifts)

Thresholding: M' = (φ' < 0)
```

### Advantages of SDF-Based Augmentation

| Property | Pixel-Level | SDF-Based |
|----------|-------------|-----------|
| Boundary smoothness | May produce artifacts | Always smooth |
| Anatomical plausibility | Variable | High |
| Explicit displacement control | No | Yes (in pixels) |
| Topological consistency | May break | Preserved |
| Computational cost | Lower | Moderate |

### SDF Operations

1. **Global Offset**: `φ' = φ + c`
   - c > 0: Uniform erosion (shrinkage)
   - c < 0: Uniform dilation (expansion)

2. **Gaussian Random Field Perturbation**:
   - Low-frequency: Smooth, spatially correlated boundary shifts
   - Correlation length: Controls smoothness of variations
   - Amplitude: Controls magnitude of boundary displacement

3. **Regularization**:
   - Total Variation: Limits boundary roughness
   - Area constraint: Prevents excessive size changes

## Usage

### Generate Augmented Data (Pixel-Level)

```bash
python scripts/generate_augmented_data.py \
    --data_root ./dataset/processed \
    --output_dir ./dataset/augmented \
    --dataset BUSI \
    --target_samples 100000 \
    --seed 42
```

### Generate Augmented Data (SDF-Based, Recommended)

```bash
python scripts/generate_augmented_data.py \
    --data_root ./dataset/processed \
    --output_dir ./dataset/augmented \
    --dataset BUSI \
    --target_samples 100000 \
    --seed 42 \
    --use_sdf
```

### Load Augmented Data for Training

```python
from data import get_augmented_dataloaders

train_loader, val_loader = get_augmented_dataloaders(
    data_root='./dataset/augmented',
    dataset_name='BUSI',
    batch_size=4,
    img_size=1024,
)

# Each batch contains:
# - 'image': (B, 3, 1024, 1024) - Original image
# - 'label': (B, 1024, 1024) - Ground truth mask
# - 'coarse_mask': (B, 1024, 1024) - Augmented coarse mask
# - 'dice': List of actual Dice scores
```

### Curriculum Learning

```python
from data import CurriculumAugmentedDataset

dataset = CurriculumAugmentedDataset(
    data_root='./dataset/augmented',
    dataset_name='BUSI',
)

# Training loop with curriculum
for epoch in range(total_epochs):
    dataset.update_difficulty(epoch, total_epochs)
    # Difficulty increases: easy samples (high Dice) → hard samples (low Dice)
```

## Statistics

After generation, the metadata.json contains:

```json
{
  "dataset": "BUSI",
  "original_samples": 3000,
  "target_samples": 100000,
  "statistics": {
    "total_samples": 99000,
    "dice_distribution": {
      "low": 29700,
      "medium": 49500,
      "high": 19800
    },
    "dice_mean": 0.832,
    "dice_std": 0.098,
    "augmentation_stats": {
      "erosion": 45000,
      "dilation": 42000,
      "holes": 38000,
      "elastic": 35000,
      "noise": 30000,
      "edge_roughening": 25000,
      "partial_dropout": 22000,
      "false_positives": 20000
    }
  }
}
```

## Design Rationale

### Why These Augmentation Types?

1. **Boundary-based failures** (erosion, dilation, elastic) are the most common in real segmentation models due to:
   - Feature map resolution loss in encoder-decoder architectures
   - Upsampling artifacts
   - Spatial inconsistency from attention mechanisms

2. **Region-based failures** (holes, dropout) simulate:
   - Attention dropout
   - Occlusion handling failures
   - Multi-scale feature aggregation issues

3. **Detection failures** (false positives) simulate:
   - Texture confusion
   - Similar structure misidentification
   - Noise sensitivity

### Why This Distribution?

- **30% Low Dice (0.6-0.8)**: Enough hard examples to learn refinement for severe cases
- **50% Medium Dice (0.8-0.9)**: Most real-world predictions fall here
- **20% High Dice (0.9+)**: Prevent over-correction, maintain precision

This distribution matches observed TransUNet performance across medical imaging datasets.

## Integration with Training Pipeline

```
                    Training Flow

┌─────────────────────────────────────────────────────┐
│  Phase 1: TransUNet Pre-training (existing)         │
│  Input: Images + GT Masks                           │
│  Output: Trained TransUNet                          │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Data Generation: This Augmentation System          │
│  Input: GT Masks                                    │
│  Output: 100K (Image, GT, Coarse) triplets         │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Phase 2: SAM Finetuning with Augmented Data        │
│  Input: Images + Coarse Masks (augmented)           │
│  Supervision: GT Masks                              │
│  Output: Finetuned SAM                              │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Phase 3: End-to-End Joint Training                 │
│  Input: Images                                      │
│  TransUNet → SAMRefiner                             │
│  Output: Final UltraRefiner Model                   │
└─────────────────────────────────────────────────────┘
```
