# UltraRefiner: End-to-End Differentiable Segmentation Refinement with Gradient-Coupled Foundation Models

## Abstract

Medical image segmentation remains a fundamental challenge in computer-aided diagnosis, particularly for breast lesion segmentation in ultrasound imaging where low contrast, speckle noise, ill-defined boundaries, and heterogeneous tumor morphology create significant obstacles. While deep learning architectures such as TransUNet (combining CNN locality with Vision Transformer global attention) achieve reasonable segmentation performance, they frequently produce masks with imprecise boundaries, topological errors (holes, bridges, fragmentation), and catastrophic failures on ambiguous cases. The Segment Anything Model (SAM), trained on over 1 billion masks, offers powerful zero-shot segmentation and refinement capabilities through its prompt-based architecture. However, SAM cannot be naively integrated into a segmentation pipeline because: (1) it requires carefully designed prompts (points, boxes, masks) that must be extracted from upstream predictions, and (2) standard prompt extraction methods (argmax, thresholding) are non-differentiable, preventing end-to-end optimization.

**UltraRefiner** addresses these fundamental limitations by introducing a **fully differentiable two-stage segmentation refinement framework** that enables gradient flow from the final refined output through SAM's prompt encoding back to the coarse segmentation network. This end-to-end differentiability allows joint optimization of both networks, where TransUNet learns to produce outputs that are optimally suited for SAM refinement, while SAM learns to correct the specific failure modes of the upstream network.

### Core Technical Contributions

#### 1. Differentiable Prompt Extraction from Soft Probability Masks

The key innovation enabling end-to-end training is a suite of **fully differentiable prompt extraction mechanisms** that operate on soft probability masks (continuous values in [0,1]) rather than binary predictions:

**Point Prompts via Soft-Argmax**: The foreground centroid is computed as a probability-weighted spatial expectation:
```
x_fg = Σ(P(i,j) × x_j) / Σ(P(i,j))
y_fg = Σ(P(i,j) × y_i) / Σ(P(i,j))
```
where P(i,j) is the soft mask probability at location (i,j). The background point is extracted from the inverse mask within the predicted bounding box region, ensuring it lies in a contextually relevant area. Gradients flow through these weighted averages: ∂center/∂mask = (coord - center) / mask_sum.

**Box Prompts via Weighted Statistics**: Bounding boxes are computed using mask-weighted coordinate statistics rather than hard thresholding:
```
center = Σ(P × coord) / Σ(P)
std = √(Σ(P × (coord - center)²) / Σ(P))
box = [center - 2.5×std, center + 2.5×std]
```
The 2.5σ coverage ensures ~99% of the predicted region is enclosed. This weighted approach correctly restricts computation to high-probability regions, unlike soft-min/max methods that can be dominated by low-probability background pixels.

**Mask Prompts via Adaptive Conversion**: Soft probability masks are converted to SAM's expected logit format through two strategies:
- **Direct style** (for Phase 2-finetuned SAM): `logits = (P × 2 - 1) × 10`, mapping [0,1] → [-10,+10]
- **Gaussian style** (for unfinetuned SAM): Apply adaptive Gaussian blur before conversion to create softer boundaries that match SAM's training distribution

#### 2. Three-Phase Curriculum Training Strategy

UltraRefiner employs a carefully designed **three-phase training curriculum** that progressively builds the joint system:

**Phase 1 - Coarse Network Training**: TransUNet is trained independently using standard segmentation losses (Cross-Entropy + Dice) with K-fold cross-validation. This establishes a strong baseline and produces the characteristic failure modes that Phase 2 must address.

**Phase 2 - SAM Distribution Alignment**: This critical phase bridges the distribution gap between TransUNet's soft probability outputs and SAM's training distribution (binary masks with sharp boundaries). A novel **mask augmentation system** with **12 primary error types** simulates realistic TransUNet failure patterns:

| Error Type | Description | Purpose |
|------------|-------------|---------|
| Over-segmentation (1.2x-20x) | Morphological dilation | Teach boundary contraction |
| Under-segmentation (0.4x-0.9x) | Morphological erosion | Teach boundary expansion |
| Missing chunks (5-30%) | Wedge/blob cutouts | Teach region completion |
| Internal holes (2-20% each) | Random interior voids | Teach hole filling |
| Bridge/adhesion artifacts | Thin connecting bands | Teach artifact removal |
| False positive islands | Scattered spurious blobs | Teach false positive suppression |
| Fragmentation | Cuts through mask | Teach fragment merging |
| Spatial shift (5-30%) | Translation errors | Teach location correction |
| Empty/noise predictions | Complete failures | Teach recovery from catastrophic errors |

Each augmented mask undergoes **soft mask conversion** using signed distance transform with random temperature sampling:
```
signed_dist = distance_inside - distance_outside
P(x) = sigmoid(signed_dist(x) / temperature)
```
where temperature ∈ [2.0, 8.0] controls boundary sharpness, matching TransUNet's output characteristics.

**Phase 2 Quality-Aware Loss**: A novel loss function prevents SAM from "correcting" regions that the coarse mask already got right:
```
L = L_BCE + L_Dice + λ × L_change_penalty
L_change_penalty = ||SAM_output - coarse||² × I(coarse == GT)
```
where I(·) indicates correct prediction regions. This teaches SAM to preserve high-quality inputs while aggressively refining erroneous regions.

**Phase 3 - End-to-End Joint Optimization**: Both networks are trained jointly with gradients flowing from the refined loss through differentiable prompts to TransUNet. Critical protection mechanisms prevent TransUNet degradation:
- **Gradient scaling**: Scale gradients to TransUNet by factor α ∈ [0.01, 0.1]
- **Weight regularization**: L2 penalty anchoring weights to Phase 1: `L_reg = β||W - W_phase1||²`
- **Two-stage unfreezing**: Optionally freeze TransUNet for initial epochs while SAM adapts
- **Dual loss supervision**: `L = λ_coarse × L(TransUNet) + λ_refined × L(SAM)` with λ_coarse > λ_refined

#### 3. Differentiable ROI Cropping for Resolution Enhancement

A **fully differentiable ROI cropping module** focuses SAM computation on the lesion region at maximum resolution:

1. **Soft box extraction**: Compute weighted bounding box from coarse mask (center ± 2.5σ)
2. **Context expansion**: Expand box by configurable ratio (default 20%) for surrounding context
3. **Differentiable crop**: Use `F.grid_sample` with bilinear interpolation (preserves gradients)
4. **Resolution upscaling**: Resize cropped ROI to 1024×1024 (SAM's native resolution)
5. **SAM refinement**: Process at full resolution for maximum boundary detail
6. **Differentiable paste-back**: Inverse `grid_sample` to restore refined mask to original coordinates

This pipeline is fully differentiable, allowing gradients to flow through crop coordinates back to the coarse mask. The key benefit is that lesions (often small relative to the full image) are processed at 4-16× effective resolution compared to whole-image processing.

#### 4. Gated Residual Refinement (Alternative Architecture)

For scenarios where Phase 2 finetuning is impractical, UltraRefiner offers a **gated residual refinement** architecture that constrains SAM to act as a controlled error corrector:

```
final = coarse + gate × (SAM - coarse)
      = (1 - gate) × coarse + gate × SAM
```

The gate is computed from **coarse mask uncertainty**:
```
confidence = |2 × coarse - 1|  ∈ [0, 1]
uncertainty = 1 - confidence^γ
gate = gate_min + (gate_max - gate_min) × uncertainty
```

When coarse ≈ 0.5 (uncertain), gate → gate_max, trusting SAM's correction. When coarse ≈ 0 or 1 (confident), gate → gate_min, preserving the original prediction. This architecture prevents unfinetuned SAM from degrading accurate predictions while still allowing targeted corrections in ambiguous regions.

Three gate variants are supported:
- **Uncertainty gate**: No learnable parameters, based purely on coarse confidence
- **Learned gate**: Small CNN (3 conv layers, ~3K params) predicts correction regions
- **Hybrid gate**: Product of uncertainty and learned gates

#### 5. Model Architecture Details

**TransUNet Backbone** (Coarse Segmentation):
- **Encoder**: ResNet50 (pretrained) → 12-layer ViT-B/16 transformer
- **Decoder**: Progressive upsampling with skip connections from ResNet stages
- **Input**: 224×224 grayscale/RGB images
- **Output**: 2-channel logits (background, foreground) → softmax → soft probability mask

**SAM Refiner** (Fine Segmentation):
- **Image Encoder**: ViT-B/16 or ViT-H (typically frozen to preserve pretrained representations)
- **Prompt Encoder**: Learnable embeddings for points, boxes, masks (trainable in Phases 2-3)
- **Mask Decoder**: Transformer-based decoder with IoU prediction head (trainable)
- **Multi-mask output**: 3 candidate masks with IoU scores, combined via soft IoU-weighted selection:
  ```
  weights = softmax(IoU_predictions / τ)
  refined = Σ(mask_i × weight_i)
  ```

**End-to-End Pipeline**:
```
Image (224×224)
  → TransUNet → Coarse Mask (soft, 224×224)
  → Upscale to 1024×1024
  → [Optional: ROI Crop]
  → Extract Points, Box, Mask (differentiable)
  → SAM Prompt Encoder
  → SAM Image Encoder (may be cached)
  → SAM Mask Decoder
  → Multi-mask selection (soft IoU-weighted)
  → [Optional: ROI Paste-back]
  → Refined Mask (1024×1024 or 224×224)
```

#### 6. Loss Functions

**Phase 1** (TransUNet only):
```
L = 0.5 × CrossEntropy(logits, GT) + 0.5 × Dice(softmax(logits), GT)
```

**Phase 2** (SAM finetuning with quality-aware loss):
```
L = BCE(σ(SAM), GT) + Dice(σ(SAM), GT) + λ × ChangePenalty
ChangePenalty = mean(|σ(SAM) - coarse| × I(coarse_binary == GT))
```

**Phase 3** (End-to-end):
```
L = λ_c × L_coarse(TransUNet, GT) + λ_r × L_refined(SAM, GT) + λ_w × ||W - W_phase1||²
```
Default: λ_c = 0.5-0.8, λ_r = 0.2-0.5, λ_w = 0.01-0.1

#### 7. Data Pipeline Optimizations

**Offline Augmentation** (Recommended): Pre-generate augmented masks to eliminate CPU bottleneck:
- 5× data multiplier per original sample
- Metadata.json for O(1) sample indexing (no directory scanning)
- File copying (not symlinks) for fast I/O on network filesystems
- Mixed precision (AMP) for 2-4× training speedup

**Hybrid Training**: Combine real TransUNet predictions with augmented GT:
- Out-of-fold prediction strategy (each fold's model predicts on its validation set)
- Configurable mixing ratio (e.g., 70% real, 30% augmented)
- Captures actual failure modes while maintaining diversity

### Summary

UltraRefiner represents a principled approach to integrating foundation models (SAM) with task-specific networks (TransUNet) in a fully differentiable manner. The key innovations are:

1. **Differentiable prompt extraction** enabling gradient flow through soft-argmax points, weighted-statistics boxes, and direct/Gaussian mask conversion
2. **Three-phase curriculum** progressively building from independent training to distribution alignment to joint optimization
3. **12-type mask augmentation** comprehensively simulating coarse network failure modes with soft mask conversion
4. **Quality-aware loss** teaching SAM to selectively refine while preserving accurate predictions
5. **Differentiable ROI cropping** for resolution-enhanced lesion processing
6. **Gated residual refinement** constraining corrections to uncertain regions
7. **TransUNet protection mechanisms** preventing performance degradation during end-to-end training

The framework is designed for medical image segmentation but generalizes to any domain requiring refinement of coarse segmentation predictions.

---

## Architecture

```
+-----------------------------------------------------------------------------+
|                    UltraRefiner Pipeline (with ROI Cropping)                 |
+-----------------------------------------------------------------------------+
|                                                                              |
|   Input Image (224x224)                                                      |
|         |                                                                    |
|         v                                                                    |
|   +-------------------------------------------------------------------+     |
|   |                         TransUNet                                  |     |
|   |     ResNet50 --> ViT-B/16 Transformer --> CNN Decoder + Skip       |     |
|   +-----------------------------+-----------------------------------------+  |
|                                 |                                            |
|                                 v                                            |
|                    Coarse Mask (Soft Probability)                            |
|                          P(lesion) in [0, 1]                                 |
|                                 |                                            |
|              +------------------+------------------+                         |
|              |                  |                  |                         |
|              v                  v                  v                         |
|        +----------+      +-----------+      +----------+                    |
|        |  Points  |      |    Box    |      |   Mask   |                    |
|        | soft-    |      | weighted  |      |  direct  |                    |
|        | argmax   |      | mean+std  |      | (logits) |                    |
|        +----+-----+      +-----+-----+      +----+-----+                    |
|             |                  |                  |                          |
|             +------------------+------------------+                          |
|                                |                                             |
|   +----------------------------+----------------------------+               |
|   |              Differentiable ROI Cropper (Default)       |               |
|   +----------------------------+----------------------------+               |
|   |                                                         |               |
|   |   1. Compute soft bounding box from mask (center + 2.5s)|               |
|   |   2. Expand box by 20% (roi_expand_ratio=0.2)           |               |
|   |   3. Crop image & mask via grid_sample (differentiable) |               |
|   |   4. Resize ROI to 1024x1024 (full SAM resolution)      |               |
|   |                                                         |               |
|   +----------------------------+----------------------------+               |
|                                |                                             |
|                                v                                             |
|   +-------------------------------------------------------------------+     |
|   |                        SAM Refiner                                 |     |
|   |     Image Encoder --> Prompt Encoder --> Mask Decoder              |     |
|   |       (frozen)         (trainable)        (trainable)              |     |
|   +-----------------------------+-------------------------------------+     |
|                                 |                                            |
|                                 v                                            |
|                     Refined Mask (ROI space)                                 |
|                                 |                                            |
|   +----------------------------+----------------------------+               |
|   |              Differentiable Paste Back                  |               |
|   |   grid_sample inverse: ROI mask -> Original image space |               |
|   +----------------------------+----------------------------+               |
|                                 |                                            |
|                                 v                                            |
|                         Final Refined Mask                                   |
|                                                                              |
|   <--- Gradient Flow: L_refined -> Paste -> SAM -> Crop -> Prompts -> TU    |
|                                                                              |
+-----------------------------------------------------------------------------+
```

### Gated Residual Refinement Architecture

```
+---------------------------------------------------------------------+
|                      Gated Residual Refinement                       |
+---------------------------------------------------------------------+
|                                                                      |
|   coarse_mask (from TransUNet)                                       |
|        |                                                             |
|        +-------------------+                                         |
|        |                   |                                         |
|        v                   v                                         |
|   +---------+       +--------------+                                |
|   |   SAM   |       | Uncertainty  |                                |
|   | Refiner |       |    Gate      |                                |
|   +----+----+       | 1-|2p-1|^g   |                                |
|        |            +------+-------+                                |
|        v                   |                                         |
|   sam_output               |                                         |
|        |                   |                                         |
|        +----------+--------+                                         |
|                   |                                                  |
|                   v                                                  |
|   +-------------------------------------------------------+         |
|   |  final = coarse + gate x (sam_output - coarse)        |         |
|   |                                                        |         |
|   |  gate ~ 0 (confident): final ~ coarse (preserve)       |         |
|   |  gate ~ 1 (uncertain): final ~ sam    (correct)        |         |
|   +-------------------------------------------------------+         |
|                                                                      |
+---------------------------------------------------------------------+
```

---

## Training Pipeline

### Phase 1: TransUNet Training

Train coarse segmentation independently with 5-fold cross-validation.

```bash
python scripts/train_transunet.py \
    --data_root ./dataset/processed \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --fold 0 \
    --vit_pretrained ./pretrained/R50+ViT-B_16.npz \
    --img_size 224 \
    --batch_size 24 \
    --max_epochs 150
```

---

### Phase 2: SAM Finetuning (Distribution Alignment)

Phase 2 is **critical** for aligning SAM with TransUNet's output distribution.

#### Option A: Offline Augmentation (RECOMMENDED)

Pre-generate augmented masks for fast, reproducible training:

**Step 1: Generate Augmented Masks**
```bash
python scripts/generate_augmented_masks.py \
    --data_root ./dataset/processed \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --output_dir ./dataset/augmented_masks \
    --num_augmentations 5 \
    --augmentor_preset default \
    --num_workers 16
```

**Step 2: Train SAM**
```bash
python scripts/finetune_sam_offline.py \
    --data_root ./dataset/augmented_masks \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --sam_checkpoint ./pretrained/sam_vit_b_01ec64.pth \
    --output_dir ./checkpoints/sam_finetuned \
    --mask_prompt_style direct \
    --transunet_img_size 224 \
    --use_roi_crop \
    --roi_expand_ratio 0.3 \
    --use_amp \
    --batch_size 4 \
    --epochs 100
```

#### Option B: Hybrid Training (Real + Augmented)

Mix real TransUNet predictions with augmented GT:

```bash
# Step 1: Generate out-of-fold predictions
python scripts/generate_transunet_predictions.py \
    --data_root ./dataset/processed \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --checkpoint_dir ./checkpoints/transunet \
    --output_dir ./dataset/transunet_preds

# Step 2: Train with hybrid data
python scripts/finetune_sam_offline.py \
    --data_root ./dataset/augmented_masks \
    --pred_data_root ./dataset/transunet_preds \
    --real_ratio 0.7 \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --sam_checkpoint ./pretrained/sam_vit_b_01ec64.pth \
    --output_dir ./checkpoints/sam_finetuned_hybrid \
    --mask_prompt_style direct \
    --transunet_img_size 224 \
    --use_roi_crop \
    --roi_expand_ratio 0.3 \
    --use_amp \
    --batch_size 4 \
    --epochs 100
```

| `--real_ratio` | Meaning |
|----------------|---------|
| `0.0` | 100% augmented (default) |
| `0.7` | 70% real TransUNet predictions, 30% augmented |
| `0.5` | 50% each |

---

### Phase 3: End-to-End Training

Joint optimization with gradient flow through differentiable prompts.

```bash
python scripts/train_e2e.py \
    --data_root ./dataset/processed \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --transunet_checkpoint ./checkpoints/transunet/combined/fold_0/best.pth \
    --sam_checkpoint ./checkpoints/sam_finetuned/best_sam.pth \
    --fold 0 \
    --n_splits 5 \
    --mask_prompt_style direct \
    --use_roi_crop \
    --roi_expand_ratio 0.3 \
    --batch_size 4 \
    --max_epochs 100 \
    --transunet_lr 1e-5 \
    --sam_lr 1e-5 \
    --coarse_loss_weight 0.5 \
    --refined_loss_weight 0.5 \
    --transunet_grad_scale 0.1 \
    --transunet_weight_reg 0.01 \
    --output_dir ./checkpoints/ultra_refiner
```

#### Critical Phase 2 -> Phase 3 Consistency

| Setting | Phase 2 | Phase 3 | Reason |
|---------|---------|---------|--------|
| `mask_prompt_style` | `direct` | `direct` | Same mask preprocessing |
| `transunet_img_size` | `224` | `224` | Same resolution path |
| `use_roi_crop` | `True` | `True` | Same cropping behavior |
| `roi_expand_ratio` | `0.3` | `0.3` | Same crop boundaries |

#### TransUNet Protection Options

| Flag | Recommended | Purpose |
|------|-------------|---------|
| `--transunet_grad_scale` | `0.1` | Scale gradients from SAM (10%) |
| `--transunet_weight_reg` | `0.01` | L2 penalty to anchor weights |
| `--freeze_transunet_epochs` | `5` | Let SAM adapt first |
| `--coarse_loss_weight` | `0.5-0.8` | Maintain TransUNet quality |

---

### Evaluation on Test Sets

Evaluate the trained model on test sets of each dataset, showing both coarse (TransUNet) and refined (SAM) performance:

```bash
# Basic evaluation (at 224x224 resolution)
python scripts/evaluate_test.py \
    --checkpoint ./checkpoints/ultra_refiner/fold_0/best.pth \
    --data_root ./dataset/processed \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM

# Evaluate at SAM's native resolution (1024x1024) to preserve boundary details
python scripts/evaluate_test.py \
    --checkpoint ./checkpoints/ultra_refiner/fold_0/best.pth \
    --data_root ./dataset/processed \
    --refined_eval_size 1024

# For gated refinement models
python scripts/evaluate_test.py \
    --checkpoint ./checkpoints/ultra_refiner/fold_0/best.pth \
    --data_root ./dataset/processed \
    --use_gated_refinement \
    --gate_type uncertainty
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--refined_eval_size` | `224` | Resolution for SAM evaluation (224 or 1024) |
| `--datasets` | All | Specific datasets to evaluate |
| `--output_dir` | `./results/test_evaluation` | Where to save JSON results |

**Output**: Detailed metrics (Dice, IoU, Precision, Recall, Accuracy) for each dataset with improvement comparison between coarse and refined outputs.

---

## Mask Augmentation System

### 12 Primary Error Types

| # | Error Type | Probability | Description |
|---|------------|-------------|-------------|
| 1 | Identity | 15% | Preserve good predictions (0-3px change) |
| 2 | Over-Segmentation | 17% | 1.2x-3x area expansion |
| 3 | Giant Over-Segmentation | 10% | 3x-20x area expansion |
| 4 | Under-Segmentation | 17% | 0.4x-0.9x area shrinkage |
| 5 | Missing Chunk | 12% | 5-30% wedge/blob cutout |
| 6 | Internal Holes | 10% | 1-3 holes (2-20% each) |
| 7 | Bridge/Adhesion | 10% | Thin band artifact + FP blob |
| 8 | False Positive Islands | 17% | 1-30 scattered blobs |
| 9 | Fragmentation | 10% | 1-5 cuts through mask |
| 10 | Shift/Location | 10% | 5-30% translation |
| 11 | Empty Prediction | 4% | Complete miss |
| 12 | Noise Scatter | 3% | Pure false positive noise |

### Augmentor Presets

| Preset | Description |
|--------|-------------|
| `default` | Balanced distribution (recommended) |
| `mild` | More identity (30%), fewer severe errors |
| `severe` | More extreme failures (giant, empty, scatter) |
| `boundary_focus` | 50% over/under-segmentation |
| `structural` | 40% holes + missing + fragmentation |

---

## Gated Residual Refinement

For scenarios where Phase 2 is impractical, use gated refinement:

```bash
python scripts/train_e2e.py \
    --data_root ./dataset/processed \
    --transunet_checkpoint ./checkpoints/transunet/best.pth \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --use_gated_refinement \
    --gate_type uncertainty \
    --gate_gamma 1.0 \
    --gate_max 0.5 \
    --mask_prompt_style gaussian \
    --use_roi_crop
```

| Gate Type | Description | Params |
|-----------|-------------|--------|
| `uncertainty` | Based on coarse confidence: `1-|2p-1|^γ` | None |
| `learned` | CNN predicts correction regions | ~3K |
| `hybrid` | Product of both | ~3K |

---

## Project Structure

```
UltraRefiner/
+-- models/
|   +-- ultra_refiner.py          # UltraRefiner & GatedUltraRefiner
|   +-- sam_refiner.py            # DifferentiableSAMRefiner + ROI Cropper
|   +-- transunet/                # Vision Transformer backbone
|   +-- lora.py                   # Low-rank adaptation
+-- data/
|   +-- dataset.py                # K-fold data loaders
|   +-- mask_augmentation.py      # 12 error types + soft mask conversion
|   +-- offline_augmented_dataset.py   # Offline augmentation (recommended)
|   +-- offline_hybrid_dataset.py      # Hybrid (real + augmented)
|   +-- online_augmented_dataset.py    # Online augmentation (slower)
+-- scripts/
|   +-- train_transunet.py        # Phase 1
|   +-- generate_augmented_masks.py    # Generate offline data
|   +-- generate_transunet_predictions.py  # Out-of-fold predictions
|   +-- finetune_sam_offline.py   # Phase 2 (recommended)
|   +-- train_e2e.py              # Phase 3
|   +-- evaluate_test.py          # Test set evaluation
|   +-- inference.py              # Inference pipeline
+-- utils/
|   +-- losses.py                 # Dice, BCE, quality-aware losses
|   +-- metrics.py                # Evaluation metrics & logging
+-- configs/
    +-- config.py                 # Project configuration
```

---

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.0

```bash
pip install -r requirements.txt

# Download pretrained weights
wget https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz -P ./pretrained/
# Download SAM from https://github.com/facebookresearch/segment-anything
# Download MedSAM from https://github.com/bowang-lab/MedSAM
```

---

## Citation

```bibtex
@article{ultrarefiner2024,
  title={UltraRefiner: End-to-End Differentiable Segmentation Refinement with Gradient-Coupled Foundation Models},
  author={},
  year={2024}
}
```

## References

- [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306)
- [Segment Anything](https://segment-anything.com/)
- [MedSAM: Segment Anything in Medical Images](https://github.com/bowang-lab/MedSAM)
