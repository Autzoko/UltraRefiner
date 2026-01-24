# UltraRefiner: End-to-End Differentiable Segmentation Refinement

## Abstract

Accurate segmentation of breast lesions in ultrasound images is critical for computer-aided diagnosis, yet existing deep learning approaches often produce masks with imprecise boundaries, topological errors, or complete failures on challenging cases. We present **UltraRefiner**, an end-to-end differentiable framework that combines a coarse segmentation network with the Segment Anything Model (SAM) for iterative mask refinement. Unlike conventional cascaded approaches where refinement modules are trained independently, UltraRefiner enables gradient flow from the refinement stage back to the coarse segmentation network, allowing joint optimization of the entire pipeline.

**Key Contributions:**

1. **Fully Differentiable Prompt Generation**: We introduce differentiable methods for extracting SAM prompts (points, boxes, and masks) from soft probability maps using soft-argmax for centroid extraction and soft-min/max with temperature scaling for bounding box computation. This enables end-to-end gradient flow through the entire pipeline.

2. **SDF-Based Failure Simulation**: We propose a signed distance function (SDF) based data augmentation strategy that generates realistic segmentation failures—including boundary erosion, dilation, topological errors, and small-lesion disappearance—with controlled Dice score distribution. This enables training the refinement module on diverse failure patterns without requiring actual model predictions.

3. **Quality-Aware Training Loss**: We design a change-penalty loss that learns "when not to modify" by penalizing unnecessary changes to already-good predictions while encouraging strong corrections on poor inputs. This prevents the refinement module from degrading high-quality inputs.

4. **Differentiable ROI Cropping**: We implement fully differentiable region-of-interest cropping using grid sampling, allowing SAM to process lesion regions at full resolution while maintaining gradient flow for end-to-end training.

5. **Phase-Consistent Distribution Matching**: We ensure that the coarse mask distribution seen during SAM finetuning matches the distribution during end-to-end training through soft probability maps and consistent resolution paths (224×224 → 1024×1024).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           UltraRefiner Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Input Image ──────────────────────────────────────────────────────────    │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    TransUNet (224×224)                               │   │
│   │   ┌──────────┐     ┌──────────────┐     ┌──────────────┐            │   │
│   │   │ ResNet50 │ ──► │  ViT-B/16    │ ──► │ CNN Decoder  │            │   │
│   │   │ Encoder  │     │ Transformer  │     │ + Skip Conn  │            │   │
│   │   └──────────┘     └──────────────┘     └──────────────┘            │   │
│   └──────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                           │
│                                  ▼                                           │
│                    ┌─────────────────────────────┐                          │
│                    │     Coarse Mask (Soft)      │                          │
│                    │   P(lesion) ∈ [0, 1]        │                          │
│                    └─────────────┬───────────────┘                          │
│                                  │                                           │
│         ┌────────────────────────┼────────────────────────┐                 │
│         │                        │                        │                 │
│         ▼                        ▼                        ▼                 │
│   ┌───────────┐          ┌───────────────┐         ┌───────────┐           │
│   │  Points   │          │    Boxes      │         │   Mask    │           │
│   │ soft-argmax│         │ soft-min/max  │         │  direct   │           │
│   └─────┬─────┘          └───────┬───────┘         └─────┬─────┘           │
│         │                        │                        │                 │
│         └────────────────────────┼────────────────────────┘                 │
│                                  │                                           │
│                                  ▼                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      SAMRefiner (1024×1024)                          │   │
│   │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │   │
│   │   │    Image     │    │   Prompt     │    │     Mask     │          │   │
│   │   │   Encoder    │    │   Encoder    │ ─► │    Decoder   │          │   │
│   │   │  (frozen)    │    │ (trainable)  │    │  (trainable) │          │   │
│   │   └──────────────┘    └──────────────┘    └──────────────┘          │   │
│   └──────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                           │
│                                  ▼                                           │
│                    ┌─────────────────────────────┐                          │
│                    │     Refined Mask            │                          │
│                    │   (High-resolution output)  │                          │
│                    └─────────────────────────────┘                          │
│                                                                              │
│   Gradient Flow: L_refined ──► SAMRefiner ──► Prompts ──► TransUNet        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Training Pipeline

UltraRefiner employs a three-phase training strategy:

| Phase | Description | What's Trained | Data Used |
|-------|-------------|----------------|-----------|
| **Phase 1** | Train TransUNet independently | TransUNet | Original images + GT masks |
| **Phase 2** | Finetune SAM with failure simulation | SAM (prompt encoder + mask decoder) | Augmented coarse masks |
| **Phase 3** | End-to-end joint optimization | TransUNet + SAM | Original images + GT masks |

**Phase 1: Coarse Segmentation Training**
- Train TransUNet at 224×224 resolution using standard segmentation losses
- Per-dataset training with 5-fold cross-validation
- Produces baseline coarse segmentation capability

**Phase 2: Refinement Module Training**
- Generate large-scale augmented data (~100K samples) from GT masks
- SDF-based failure simulation creates realistic segmentation errors
- Quality distribution: 25% good (Dice 0.9-0.99), 40% medium (0.8-0.9), 35% poor (0.6-0.8)
- No perfect masks (Dice=1.0) to ensure the refiner learns to modify
- Quality-aware loss penalizes changes to already-good inputs

**Phase 3: End-to-End Training**
- Load Phase 1 TransUNet and Phase 2 SAM checkpoints
- Joint optimization with gradients flowing through differentiable prompts
- Combined loss: L = 0.3 × L_coarse + 0.7 × L_refined
- Lower learning rate for TransUNet to prevent destabilization

## Phase 2 ↔ Phase 3 Distribution Consistency

For the finetuned SAM from Phase 2 to work correctly in Phase 3, the input distributions must match exactly. Here's how consistency is maintained:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTION CONSISTENCY DIAGRAM                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 2 (SAM Finetuning)              PHASE 3 (E2E Training)               │
│  ─────────────────────────             ─────────────────────────             │
│                                                                              │
│  Original Image                        Original Image                        │
│       │                                     │                                │
│       ▼                                     ▼                                │
│  ┌─────────────┐                      ┌─────────────┐                       │
│  │ Resize 224  │ ◄─── SAME ───────►   │ Resize 224  │ (for TransUNet)       │
│  └──────┬──────┘                      └──────┬──────┘                       │
│         │                                    │                               │
│         ▼                                    ▼                               │
│  ┌─────────────┐                      ┌─────────────┐                       │
│  │ Resize 1024 │ ◄─── SAME ───────►   │ Resize 1024 │ (for SAM)             │
│  └──────┬──────┘                      └──────┬──────┘                       │
│         │                                    │                               │
│         ▼                                    ▼                               │
│  ┌─────────────┐                      ┌─────────────┐                       │
│  │ SAM Normalize│ ◄─── SAME ───────►  │ SAM Normalize│                       │
│  │ (mean/std)  │                      │ (mean/std)  │                       │
│  └──────┬──────┘                      └──────┬──────┘                       │
│         │                                    │                               │
│         ▼                                    ▼                               │
│  ═══════════════                      ═══════════════                       │
│   IMAGE: 1024²                         IMAGE: 1024²                          │
│   Normalized                           Normalized                            │
│  ═══════════════                      ═══════════════                       │
│                                                                              │
│  Soft Mask (NPY)                      TransUNet Output                       │
│  (Gaussian blur)                      (Softmax prob)                         │
│       │                                     │                                │
│       ▼                                     ▼                                │
│  ┌─────────────┐                      ┌─────────────┐                       │
│  │ Resize 224  │ ◄─── SAME ───────►   │ Output 224  │                       │
│  └──────┬──────┘                      └──────┬──────┘                       │
│         │                                    │                               │
│         ▼                                    ▼                               │
│  ┌─────────────┐                      ┌─────────────┐                       │
│  │ Resize 1024 │ ◄─── SAME ───────►   │ Resize 1024 │ (F.interpolate)       │
│  │ (Bilinear)  │                      │ (Bilinear)  │                       │
│  └──────┬──────┘                      └──────┬──────┘                       │
│         │                                    │                               │
│         ▼                                    ▼                               │
│  ═══════════════                      ═══════════════                       │
│   MASK: 1024²                          MASK: 1024²                           │
│   Soft [0,1]                           Soft [0,1]                            │
│  ═══════════════                      ═══════════════                       │
│         │                                    │                               │
│         └──────────────┬─────────────────────┘                              │
│                        ▼                                                     │
│              ┌─────────────────────┐                                        │
│              │  PROMPT EXTRACTION  │                                        │
│              │  (Same Code)        │                                        │
│              ├─────────────────────┤                                        │
│              │ • soft-argmax point │                                        │
│              │ • soft-min/max box  │                                        │
│              │ • direct mask       │                                        │
│              └─────────────────────┘                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Consistency Requirements:**

| Aspect | Phase 2 | Phase 3 | Status |
|--------|---------|---------|--------|
| Image resolution | 1024×1024 | 1024×1024 | ✓ Same |
| Image normalization | SAM pixel mean/std | SAM pixel mean/std | ✓ Same |
| Resolution path | 224→1024 (bilinear) | 224→1024 (bilinear) | ✓ Same |
| Coarse mask format | Soft probability [0,1] | Softmax probability [0,1] | ✓ Same |
| Coarse mask resolution | 1024×1024 | 1024×1024 | ✓ Same |
| Prompt extraction | DifferentiableSAMRefiner | DifferentiableSAMRefiner | ✓ Same code |
| Coordinate space | 1024×1024 pixels | 1024×1024 pixels | ✓ Same |
| mask_prompt_style | 'direct' (default) | 'direct' (default) | ✓ Same |
| ROI cropping | Configurable | Configurable | Must match |

**Ensuring Consistency (Checklist):**
1. ✅ Generate Phase 2 data with `--soft_masks` flag (matches TransUNet's smooth output)
2. ✅ Use `--transunet_img_size 224` in Phase 2 (matches Phase 3 resolution path)
3. ✅ Use `--mask_prompt_style direct` in both phases
4. ✅ Use same `--use_roi_crop` and `--roi_expand_ratio` in both phases
5. ✅ Use `best_sam.pth` checkpoint (SAM-native format) for Phase 3

## Resolution Design

| Component | Resolution | Reason |
|-----------|------------|--------|
| **TransUNet** | 224×224 | Matches ViT-B/16 pretrained resolution; memory efficient |
| **SAM** | 1024×1024 | SAM's native resolution; captures fine boundary details |

**Why this design?**
- TransUNet at 224×224 provides fast, coarse segmentation with global context
- SAM at 1024×1024 refines boundaries with high precision
- The coarse-to-fine approach is more efficient than running everything at high resolution
- SAM's refinement recovers boundary details lost at lower resolution

**Changing TransUNet resolution** (e.g., to 384 or 512) requires:
1. Retraining Phase 1 with `--img_size <new_size>`
2. Updating Phase 2 with `--transunet_img_size <new_size>`
3. Updating Phase 3 with `--img_size <new_size>`

The default 224×224 is recommended as SAM's high-resolution refinement compensates for TransUNet's lower resolution.

## ROI Cropping Mode (Optional)

UltraRefiner supports **ROI (Region of Interest) cropping** that focuses SAM computation on the lesion area:

```
Standard Mode:                          ROI Cropping Mode:
┌─────────────────┐                     ┌─────────────────┐
│  Full image     │                     │     ┌─────┐     │
│  1024×1024      │         →           │     │ ROI │     │  → Crop → Process → Paste
│    ┌───┐        │                     │     └─────┘     │
│    │ L │        │                     │                 │
│    └───┘        │                     └─────────────────┘
└─────────────────┘                     Full 1024×1024 res on lesion
```

**Benefits:**
- Higher effective resolution on the lesion area
- SAM learns to focus specifically on lesion refinement
- Reduces distraction from background regions
- Fully differentiable (gradients flow through crop/paste operations)

**Usage:**
```bash
# Phase 2 with ROI cropping
python scripts/finetune_sam_augmented.py \
    --data_root ./dataset/augmented_soft \
    --dataset BUSI \
    --sam_checkpoint ./checkpoints/sam/sam_vit_b_01ec64.pth \
    --use_roi_crop \
    --roi_expand_ratio 0.2

# Phase 3 with ROI cropping (must match Phase 2 setting)
python scripts/train_e2e.py \
    --data_root ./dataset/processed \
    --datasets BUSI \
    --transunet_checkpoint ./checkpoints/transunet/BUSI/fold_0/best.pth \
    --sam_checkpoint ./checkpoints/sam_finetuned/BUSI/best_sam.pth \
    --use_roi_crop \
    --roi_expand_ratio 0.2
```

**Parameters:**
- `--use_roi_crop`: Enable ROI cropping mode
- `--roi_expand_ratio`: Ratio to expand the bounding box (0.2 = 20% expansion on each side)

**IMPORTANT:** Phase 2 and Phase 3 must use the same ROI setting for distribution consistency.

**ROI Mode Distribution Consistency:**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ROI MODE DATA FLOW                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 2 (ROI Mode)                    Phase 3 (ROI Mode)                   │
│  ─────────────────────                 ─────────────────────                 │
│                                                                              │
│  Image: 1024×1024 ────┐               Image: 224→1024 ─────┐                │
│  Mask:  1024×1024 ────┤               Mask:  224→1024 ─────┤                │
│                       ▼                                    ▼                │
│              ┌──────────────────────────────────────────────────┐           │
│              │           ROI Box Extraction                      │           │
│              │    soft-min/max from 1024×1024 mask               │           │
│              │    coords in [0, 1024] range                      │           │
│              └─────────────────┬────────────────────────────────┘           │
│                                │                                             │
│                                ▼                                             │
│              ┌──────────────────────────────────────────────────┐           │
│              │           Crop & Resize                           │           │
│              │    ROI region → 1024×1024                         │           │
│              │    (grid_sample, differentiable)                  │           │
│              └─────────────────┬────────────────────────────────┘           │
│                                │                                             │
│                                ▼                                             │
│              ┌──────────────────────────────────────────────────┐           │
│              │           Prompt Extraction                       │           │
│              │    From 1024×1024 CROPPED mask                    │           │
│              │    coords in [0, 1024] (no scaling needed)        │           │
│              └─────────────────┬────────────────────────────────┘           │
│                                │                                             │
│                                ▼                                             │
│              ┌──────────────────────────────────────────────────┐           │
│              │           SAM Processing                          │           │
│              │    1024×1024 cropped image                        │           │
│              │    Full resolution on lesion area                 │           │
│              └─────────────────┬────────────────────────────────┘           │
│                                │                                             │
│                                ▼                                             │
│              ┌──────────────────────────────────────────────────┐           │
│              │           Paste Back                              │           │
│              │    1024×1024 → original size                      │           │
│              │    (inverse affine, differentiable)               │           │
│              └──────────────────────────────────────────────────┘           │
│                                                                              │
│  KEY: All operations at 1024×1024, same coordinate space                    │
│       Prompts extracted from cropped mask (no scaling needed)               │
│       Gradients flow through crop/paste operations                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Training Pipeline Flow Chart

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        ULTRAREFINER TRAINING PIPELINE                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                     PHASE 1: TransUNet Training                         │ ║
║  │                     (Per-Dataset, 5-Fold CV)                            │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║     Raw Images ──────────────────────────────────────────────────────────    ║
║         │                                                                     ║
║         ▼                                                                     ║
║  ┌─────────────────┐    For each dataset (BUSI, BUSBRA, BUS, ...)            ║
║  │  Preprocessing  │    For each fold (0, 1, 2, 3, 4)                        ║
║  │  (224×224)      │                                                         ║
║  └─────────────────┘                                                         ║
║         │                                                                     ║
║         ▼                                                                     ║
║  ┌─────────────────────────────────────────────────────────────────┐         ║
║  │                        TransUNet                                 │         ║
║  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │         ║
║  │  │   ResNet50   │ ─► │   ViT-B/16   │ ─► │ CNN Decoder  │       │         ║
║  │  │   Encoder    │    │  Transformer │    │  + Skip Conn │       │         ║
║  │  └──────────────┘    └──────────────┘    └──────────────┘       │         ║
║  └─────────────────────────────────────────────────────────────────┘         ║
║         │                                                                     ║
║         ▼                                                                     ║
║  ┌─────────────────┐                                                         ║
║  │   Checkpoints   │   ./checkpoints/transunet/{dataset}/fold_{i}/best.pth  ║
║  └─────────────────┘                                                         ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                     PHASE 2: SAM Finetuning                             │ ║
║  │               (Using GT Mask Augmentation)                              │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║     ┌─────────────────┐         ┌─────────────────┐                         ║
║     │   GT Masks      │         │  Failure        │                         ║
║     │   (Binary)      │ ──────► │  Simulation     │                         ║
║     └─────────────────┘         └────────┬────────┘                         ║
║                                          │                                   ║
║                                          ▼                                   ║
║     ┌────────────────────────────────────────────────────────────────┐      ║
║     │              SDF-Based Augmentation Pipeline                    │      ║
║     │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │      ║
║     │  │ Mask→SDF │ ► │ Add GRF  │ ► │Threshold │ ► │ Corrupted│    │      ║
║     │  │          │   │ + Offset │   │ SDF→Mask │   │   Mask   │    │      ║
║     │  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │      ║
║     │                                                                │      ║
║     │  Failure Modes: erosion, dilation, breakage, holes,           │      ║
║     │                 bridges, small-lesion disappearance           │      ║
║     └────────────────────────────────────────────────────────────────┘      ║
║                                          │                                   ║
║     ┌─────────────────┐                  │                                   ║
║     │  Original Image │                  │                                   ║
║     │   (1024×1024)   │                  │                                   ║
║     └────────┬────────┘                  │                                   ║
║              │                           ▼                                   ║
║              │     ┌─────────────────────────────────────────┐              ║
║              │     │        Differentiable Prompt Generator  │              ║
║              │     │  ┌─────────┬──────────────┬──────────┐  │              ║
║              │     │  │  Point  │     Box      │   Mask   │  │              ║
║              │     │  └─────────┴──────────────┴──────────┘  │              ║
║              │     └─────────────────────┬───────────────────┘              ║
║              │                           │                                   ║
║              ▼                           ▼                                   ║
║  ┌─────────────────────────────────────────────────────────────────┐        ║
║  │                          SAM (MedSAM)                            │        ║
║  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │        ║
║  │  │    Image     │    │   Prompt     │    │     Mask     │       │        ║
║  │  │   Encoder    │    │   Encoder    │ ─► │    Decoder   │       │        ║
║  │  │  (FROZEN)    │    │ (trainable)  │    │  (trainable) │       │        ║
║  │  └──────────────┘    └──────────────┘    └──────────────┘       │        ║
║  └─────────────────────────────────────────────────────────────────┘        ║
║         │                                                                    ║
║         ▼                                                                    ║
║  ┌─────────────────┐   Loss = BCE-Dice + IoU Loss + Change Penalty          ║
║  │  Refined Mask   │   (Quality-aware: penalize changes on good inputs)     ║
║  └─────────────────┘                                                        ║
║         │                                                                    ║
║         ▼                                                                    ║
║  ┌─────────────────┐   ./checkpoints/sam_finetuned/{dataset}/               ║
║  │   Checkpoints   │   ├── best.pth      (full refiner, for resume)        ║
║  │                 │   └── best_sam.pth  (SAM weights, for Phase 3)        ║
║  └─────────────────┘                                                        ║
║                                                                               ║
║                              │                                                ║
║                              ▼                                                ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                  PHASE 3: End-to-End Training                           │ ║
║  │                  (Joint Optimization)                                   │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║                    ┌─────────────────────────┐                              ║
║                    │      Input Image        │                              ║
║                    │       (224×224)         │                              ║
║                    └───────────┬─────────────┘                              ║
║                                │                                             ║
║         ┌──────────────────────┴──────────────────────┐                     ║
║         ▼                                             │                     ║
║  ┌─────────────────┐                                  │                     ║
║  │    TransUNet    │  ◄── Gradients flow back ───┐    │                     ║
║  │   (trainable)   │                             │    │                     ║
║  └────────┬────────┘                             │    │                     ║
║           │                                      │    │                     ║
║           ▼                                      │    │                     ║
║  ┌─────────────────┐                             │    │                     ║
║  │   Coarse Mask   │ ──► L_coarse (CE + Dice)    │    │                     ║
║  │   (Soft Prob)   │          │                  │    │                     ║
║  └────────┬────────┘          │                  │    │                     ║
║           │                   │                  │    │                     ║
║           ▼                   │                  │    │                     ║
║  ┌─────────────────┐          │                  │    │                     ║
║  │   Differentiable│          │                  │    │                     ║
║  │     Prompts     │          │                  │    │                     ║
║  └────────┬────────┘          │                  │    │                     ║
║           │                   │                  │    │                     ║
║           ▼                   │                  │    │                     ║
║  ┌─────────────────┐          │                  │    │                     ║
║  │   SAMRefiner    │ ◄────────│──────────────────┘    │                     ║
║  │   (trainable)   │          │                       │                     ║
║  └────────┬────────┘          │                       │                     ║
║           │                   │                       │                     ║
║           ▼                   │                       │                     ║
║  ┌─────────────────┐          │                       │                     ║
║  │  Refined Mask   │ ──► L_refined (BCE-Dice) ───────┘                     ║
║  └─────────────────┘          │                                             ║
║                               ▼                                             ║
║                    L_total = 0.3 × L_coarse + 0.7 × L_refined              ║
║                                                                               ║
║                               │                                              ║
║                               ▼                                              ║
║                    ┌─────────────────┐                                      ║
║                    │   Final Model   │                                      ║
║                    │  ./checkpoints/ │                                      ║
║                    │  ultra_refiner/fold_{i}/                               ║
║                    └─────────────────┘                                      ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## Quick Start

### 1. Data Preprocessing

```bash
# Preprocess all datasets (creates train/test splits, excludes blank masks)
python scripts/preprocess_datasets.py \
    --raw_root ./dataset/raw \
    --output_root ./dataset/processed \
    --test_ratio 0.2 \
    --n_splits 5
```

### 2. Phase 1: Train TransUNet

```bash
# Train on single dataset with 5-fold CV
python scripts/train_transunet.py \
    --data_root ./dataset/processed \
    --dataset BUSI \
    --fold 0 \
    --n_splits 5 \
    --vit_pretrained ./pretrained/R50+ViT-B_16.npz \
    --max_epochs 150

# Repeat for all folds (0-4) and all datasets
```

### 3. Phase 2: Finetune SAM

**Option A: GT with Synthetic Failure Simulation (Recommended)**

This approach generates large-scale training data by simulating realistic segmentation failures on ground-truth masks, enabling the Refiner to learn both when to correct and when to preserve.

### Data Augmentation for SAM Finetuning

To train a robust SAM Refiner, we employ a systematic data augmentation strategy that simulates realistic segmentation failures. This approach expands the limited training data (~3K samples) to a large-scale dataset (~100K samples) with controlled quality distribution.

**Simulated Failure Modes:**
- **Under-segmentation**: Boundary erosion, missing parts, partial breakage
- **Over-segmentation**: Boundary dilation, attachment to nearby structures
- **Boundary roughness**: Pixel-level noise, contour jitter along edges
- **Topological errors**: Internal holes, false-positive islands, artificial bridges
- **Small-lesion failure**: Extreme shrinkage or complete disappearance (critical failure case)

**Size and Shape Conditioning:**
Augmentation intensity and failure type selection are conditioned on lesion characteristics (area, circularity, boundary complexity). Tiny or irregularly-shaped lesions receive more aggressive corruption to reflect realistic model behavior.

**SDF-Based Augmentation:**
Mask perturbations are modeled in the continuous signed distance function (SDF) domain, where the zero level-set defines the contour. Boundary shifts are expressed as additive fields (global offsets for uniform erosion/dilation, low-frequency Gaussian random fields for spatially varying deformation), ensuring smooth, anatomically plausible results. This formulation provides explicit control over boundary displacement magnitude, smoothness, and topological constraints.

**Quality-Aware Training:**
Training uses a controlled distribution with NO perfect (Dice=1.0) masks:
- **25%** Dice 0.9-0.99 (good, minor artifacts)
- **40%** Dice 0.8-0.9 (moderate errors)
- **35%** Dice 0.6-0.8 (severe failures)

A change-penalty term weighted by input mask quality encourages the Refiner to preserve already-good predictions while strongly correcting poor ones.

**IMPORTANT: Soft Masks for Phase 3 Compatibility**

For the finetuned SAM to work correctly in Phase 3 E2E training, the coarse mask distribution must match TransUNet's soft probability outputs. Use `--soft_masks` flag when generating augmented data.

#### Step 2a: Generate Augmented Data

```bash
# Generate augmented training data with SOFT MASKS
# Soft masks have smooth boundaries matching TransUNet's output distribution
python scripts/generate_augmented_data.py \
    --data_root ./dataset/processed \
    --output_dir ./dataset/augmented_soft \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --target_samples 100000 \
    --use_sdf \
    --soft_masks \
    --num_workers 8
```

#### Step 2b: Finetune SAM (RECOMMENDED Commands)

**Standard Mode (Full Image Processing):**
```bash
# RECOMMENDED: Phase 2 with aligned distribution
python scripts/finetune_sam_augmented.py \
    --data_root ./dataset/augmented_soft \
    --dataset COMBINED \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --sam_model_type vit_b \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --mask_prompt_style direct \
    --transunet_img_size 224 \
    --change_penalty_weight 0.1 \
    --output_dir ./checkpoints/sam_finetuned
```

**ROI Mode (Focused Lesion Processing):**
```bash
# Phase 2 with ROI cropping for higher effective resolution on lesions
python scripts/finetune_sam_augmented.py \
    --data_root ./dataset/augmented_soft \
    --dataset COMBINED \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --sam_model_type vit_b \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --mask_prompt_style direct \
    --transunet_img_size 224 \
    --use_roi_crop \
    --roi_expand_ratio 0.2 \
    --change_penalty_weight 0.1 \
    --output_dir ./checkpoints/sam_finetuned_roi
```

**With Curriculum Learning:**
```bash
# Curriculum learning: start with easy samples (high Dice), gradually include harder ones
python scripts/finetune_sam_augmented.py \
    --data_root ./dataset/augmented_soft \
    --dataset COMBINED \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --sam_model_type vit_b \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --mask_prompt_style direct \
    --transunet_img_size 224 \
    --curriculum \
    --change_penalty_weight 0.1 \
    --output_dir ./checkpoints/sam_finetuned
```

**Resume Training:**
```bash
# Resume from checkpoint if interrupted
python scripts/finetune_sam_augmented.py \
    --data_root ./dataset/augmented_soft \
    --dataset COMBINED \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --sam_model_type vit_b \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --mask_prompt_style direct \
    --transunet_img_size 224 \
    --output_dir ./checkpoints/sam_finetuned \
    --resume ./checkpoints/sam_finetuned/COMBINED/best.pth
```

**Phase 2 Key Parameters:**
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--mask_prompt_style` | `direct` | No extra blur (soft masks already smooth) |
| `--transunet_img_size` | `224` | Match Phase 3 resolution path |
| `--use_roi_crop` | (optional) | Focus on lesion area |
| `--roi_expand_ratio` | `0.2` | 20% expansion on each side |
| `--change_penalty_weight` | `0.1` | Penalize changes to good inputs |

**Soft Masks vs Binary Masks:**
| Feature | Binary Masks (Legacy) | Soft Masks (Recommended) |
|---------|----------------------|--------------------------|
| Format | PNG (0/255) | NPY (float 0.0-1.0) |
| Boundaries | Sharp edges | Smooth, Gaussian-blurred |
| Phase 3 compatibility | Requires `--sharpen_coarse_mask` | Direct compatibility |
| mask_prompt_style | `gaussian` (adds blur) | `direct` (no extra blur needed) |

### 4. Phase 3: End-to-End Training

**IMPORTANT:** Phase 3 settings MUST match Phase 2 for distribution consistency!

#### RECOMMENDED Commands

**Standard Mode (matches Phase 2 standard mode):**
```bash
# Phase 3 E2E training with aligned distribution
python scripts/train_e2e.py \
    --data_root ./dataset/processed \
    --datasets BUSI \
    --fold 0 \
    --n_splits 5 \
    --transunet_checkpoint ./checkpoints/transunet/BUSI/fold_0/best.pth \
    --sam_checkpoint ./checkpoints/sam_finetuned/COMBINED/best_sam.pth \
    --mask_prompt_style direct \
    --max_epochs 100 \
    --batch_size 8 \
    --transunet_lr 1e-5 \
    --sam_lr 1e-5 \
    --coarse_loss_weight 0.3 \
    --refined_loss_weight 0.7 \
    --grad_clip 1.0 \
    --output_dir ./checkpoints/ultra_refiner/BUSI
```

**ROI Mode (matches Phase 2 ROI mode):**
```bash
# Phase 3 E2E training with ROI cropping (MUST match Phase 2 ROI settings)
python scripts/train_e2e.py \
    --data_root ./dataset/processed \
    --datasets BUSI \
    --fold 0 \
    --n_splits 5 \
    --transunet_checkpoint ./checkpoints/transunet/BUSI/fold_0/best.pth \
    --sam_checkpoint ./checkpoints/sam_finetuned_roi/COMBINED/best_sam.pth \
    --mask_prompt_style direct \
    --use_roi_crop \
    --roi_expand_ratio 0.2 \
    --max_epochs 100 \
    --batch_size 8 \
    --transunet_lr 1e-5 \
    --sam_lr 1e-5 \
    --coarse_loss_weight 0.3 \
    --refined_loss_weight 0.7 \
    --grad_clip 1.0 \
    --output_dir ./checkpoints/ultra_refiner_roi/BUSI
```

**Two-Stage Training (freeze SAM initially):**
```bash
# Stage 1: Train TransUNet with frozen SAM (stabilize coarse predictions)
# Stage 2: Unfreeze SAM for joint optimization
python scripts/train_e2e.py \
    --data_root ./dataset/processed \
    --datasets BUSI \
    --fold 0 \
    --n_splits 5 \
    --transunet_checkpoint ./checkpoints/transunet/BUSI/fold_0/best.pth \
    --sam_checkpoint ./checkpoints/sam_finetuned/COMBINED/best_sam.pth \
    --mask_prompt_style direct \
    --freeze_sam_all \
    --unfreeze_sam_epoch 20 \
    --max_epochs 100 \
    --batch_size 8 \
    --transunet_lr 1e-4 \
    --sam_lr 1e-5 \
    --output_dir ./checkpoints/ultra_refiner/BUSI
```

**Without Phase 2 (Direct E2E with Original SAM):**
```bash
# Skip Phase 2, use original MedSAM directly
python scripts/train_e2e.py \
    --data_root ./dataset/processed \
    --datasets BUSI \
    --fold 0 \
    --transunet_checkpoint ./checkpoints/transunet/BUSI/fold_0/best.pth \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --mask_prompt_style direct \
    --max_epochs 100 \
    --batch_size 8 \
    --transunet_lr 1e-4 \
    --sam_lr 1e-5 \
    --output_dir ./checkpoints/ultra_refiner/BUSI
```

**Phase 3 Key Parameters:**
| Parameter | Value | Must Match Phase 2? |
|-----------|-------|---------------------|
| `--mask_prompt_style` | `direct` | ✓ YES |
| `--use_roi_crop` | `True/False` | ✓ YES |
| `--roi_expand_ratio` | `0.2` | ✓ YES (if ROI enabled) |
| `--transunet_lr` | `1e-5` | No (typically lower than sam_lr) |
| `--sam_lr` | `1e-5` | No |
| `--coarse_loss_weight` | `0.3` | No |
| `--refined_loss_weight` | `0.7` | No |

**Batch Training Script (All Folds):**
```bash
# Train all 5 folds for a dataset
for fold in 0 1 2 3 4; do
    python scripts/train_e2e.py \
        --data_root ./dataset/processed \
        --datasets BUSI \
        --fold $fold \
        --n_splits 5 \
        --transunet_checkpoint ./checkpoints/transunet/BUSI/fold_${fold}/best.pth \
        --sam_checkpoint ./checkpoints/sam_finetuned/COMBINED/best_sam.pth \
        --mask_prompt_style direct \
        --max_epochs 100 \
        --batch_size 8 \
        --transunet_lr 1e-5 \
        --sam_lr 1e-5 \
        --output_dir ./checkpoints/ultra_refiner/BUSI
done
```

**Per-Dataset Training Structure:**
```
Phase 1 TransUNet              →    Phase 3 UltraRefiner
───────────────────────             ─────────────────────
transunet/BUSI/fold_0/best.pth  →  ultra_refiner/BUSI/fold_0/best.pth
transunet/BUSI/fold_1/best.pth  →  ultra_refiner/BUSI/fold_1/best.pth
...
transunet/BUSBRA/fold_0/best.pth → ultra_refiner/BUSBRA/fold_0/best.pth
...
(5 datasets × 5 folds = 25 models)
```

**Checkpoint Format Note:**
- Phase 2 saves two checkpoint files:
  - `best.pth`: Full SAMRefiner state (for resuming training)
  - `best_sam.pth`: SAM-only weights (for Phase 3 loading)
- Phase 3 expects the SAM-native format, so always use `best_sam.pth`

### Phase 2 → Phase 3 Compatibility Checklist

| Step | Setting | Phase 2 | Phase 3 | Notes |
|------|---------|---------|---------|-------|
| 1 | Augmented data | `--soft_masks` | N/A | Generate soft probability masks |
| 2 | Mask prompt style | `--mask_prompt_style direct` | `--mask_prompt_style direct` | Must match |
| 3 | Resolution path | `--transunet_img_size 224` | Auto (224→1024) | Phase 2 simulates Phase 3 path |
| 4 | ROI cropping | `--use_roi_crop` | `--use_roi_crop` | Must match (both on or both off) |
| 5 | ROI expand ratio | `--roi_expand_ratio 0.2` | `--roi_expand_ratio 0.2` | Must match if ROI enabled |
| 6 | SAM checkpoint | Output: `best_sam.pth` | Input: `best_sam.pth` | Use SAM-native format |

**Quick Reference - Matching Configurations:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STANDARD MODE (No ROI)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Phase 2:                           Phase 3:                                │
│  --mask_prompt_style direct         --mask_prompt_style direct              │
│  --transunet_img_size 224           (auto)                                  │
│  (no --use_roi_crop)                (no --use_roi_crop)                     │
│                                                                              │
│  Output: best_sam.pth ──────────────► Input: best_sam.pth                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    ROI MODE (Focused Processing)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Phase 2:                           Phase 3:                                │
│  --mask_prompt_style direct         --mask_prompt_style direct              │
│  --transunet_img_size 224           (auto)                                  │
│  --use_roi_crop                     --use_roi_crop                          │
│  --roi_expand_ratio 0.2             --roi_expand_ratio 0.2                  │
│                                                                              │
│  Output: best_sam.pth ──────────────► Input: best_sam.pth                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why Resolution Path Matters:**
In Phase 3, TransUNet outputs at 224×224, which is then upscaled to 1024×1024 for SAM.
This upscaling creates additional smoothing. Phase 2 must simulate the same path:
- Phase 2: `original → 224×224 → 1024×1024` (with `--transunet_img_size 224`)
- Phase 3: `224×224 (TransUNet output) → 1024×1024` (SAM input)

## Project Structure

```
UltraRefiner/
├── configs/
│   └── config.py                    # Configuration management
├── models/
│   ├── transunet/                   # TransUNet model (submodule)
│   ├── sam/                         # SAM model components
│   ├── sam_refiner.py               # Differentiable SAM Refiner
│   └── ultra_refiner.py             # End-to-end model
├── data/
│   ├── dataset.py                   # Dataset loaders with K-fold CV
│   └── augmented_dataset.py         # Augmented data loaders
├── utils/
│   ├── losses.py                    # Loss functions (Dice, BCE, SAM Loss)
│   └── metrics.py                   # Metrics & TrainingLogger
├── scripts/
│   ├── preprocess_datasets.py       # Data preprocessing
│   ├── train_transunet.py           # Phase 1 training
│   ├── inference_transunet.py       # Generate predictions & visualizations
│   ├── finetune_sam.py              # Phase 2 (simulated masks)
│   ├── finetune_sam_with_preds.py   # Phase 2 (actual predictions)
│   ├── generate_augmented_data.py   # Data augmentation generation
│   ├── sdf_augmentation.py          # SDF-based augmentation
│   ├── finetune_sam_augmented.py    # Phase 2 with augmented data
│   └── train_e2e.py                 # Phase 3 training
├── dataset/
│   ├── raw/                         # Original datasets
│   ├── processed/                   # Preprocessed with splits
│   └── augmented/                   # Generated augmented training data
├── predictions/
│   └── transunet/                   # TransUNet predictions for SAM
│       └── {dataset}/fold_{i}/      # Per-dataset, per-fold predictions
├── checkpoints/
│   ├── transunet/                   # Phase 1 checkpoints
│   │   └── {dataset}/fold_{i}/      # Per-dataset, per-fold models
│   ├── sam_finetuned/               # Phase 2 checkpoints
│   │   └── fold_{i}/                # Per-fold models
│   └── ultra_refiner/               # Phase 3 checkpoints
│       └── fold_{i}/                # Per-fold models
└── pretrained/
    ├── R50+ViT-B_16.npz             # TransUNet pretrained weights
    └── medsam_vit_b.pth             # MedSAM checkpoint
```

## Datasets

| Dataset | Samples | Description |
|---------|---------|-------------|
| BUSI | 647 | Breast Ultrasound Images (benign + malignant) |
| BUSBRA | 1,875 | Brazilian Breast Ultrasound |
| BUS | 562 | Breast Ultrasound Dataset |
| BUS_UC | 956 | UC Breast Ultrasound |
| BUS_UCLM | 163 | UCLM Breast Ultrasound |

Note: Samples with blank masks (no lesion) are automatically excluded during preprocessing.

## Key Components

### Differentiable Prompt Generation

The core innovation enabling end-to-end training:

| Prompt Type | Method | Description |
|-------------|--------|-------------|
| **Point** | Soft-argmax | `centroid = Σ(mask × coords) / Σ(mask)` |
| **Box** | Threshold + MinMax | Bounding box from `mask > 0.5` |
| **Mask** | Bilinear resize | Scale to 256×256, convert to logits |

### LoRA (Low-Rank Adaptation)

UltraRefiner supports LoRA for parameter-efficient fine-tuning of SAM's image encoder:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LoRA Architecture                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Original Weight W (frozen)     LoRA Branch (trainable)         │
│  ┌─────────────────┐           ┌─────────┐   ┌─────────┐       │
│  │                 │           │  A      │   │  B      │       │
│  │   W [d × d]     │     +     │ [d × r] │ → │ [r × d] │       │
│  │                 │           │         │   │         │       │
│  └────────┬────────┘           └────┬────┘   └────┬────┘       │
│           │                         │             │             │
│           └─────────────┬───────────┴─────────────┘             │
│                         ▼                                        │
│                    Output = Wx + BAx × (α/r)                    │
│                                                                  │
│  Parameters: r << d (e.g., r=4, d=768)                          │
│  LoRA adds only ~0.3-1M params to adapt 89M frozen weights      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**LoRA Benefits:**
- Adapts frozen image encoder with minimal parameters (~0.3% of original)
- Better domain adaptation for medical images (ultrasound differs from natural images)
- Memory efficient training
- Can be merged back for inference

**Usage:**
```bash
# Finetune SAM with LoRA
python scripts/finetune_sam.py \
    --medsam_checkpoint ./pretrained/medsam_vit_b.pth \
    --use_lora \
    --lora_rank 4 \
    --lora_alpha 4.0 \
    --fold 0
```

| LoRA Parameter | Default | Description |
|----------------|---------|-------------|
| `--lora_rank` | 4 | Rank of LoRA decomposition |
| `--lora_alpha` | 4.0 | Scaling factor (typically same as rank) |
| `--lora_dropout` | 0.0 | Dropout for LoRA layers |
| `--lora_target_modules` | qkv | Which modules to apply LoRA (qkv, proj, mlp) |

### Loss Functions

- **Phase 1**: `L = 0.5 × CrossEntropy + 0.5 × Dice`
- **Phase 2**: `L = BCE-Dice + IoU_prediction_loss`
- **Phase 3**: `L = 0.3 × L_coarse + 0.7 × L_refined`

### Training Output

Beautiful console output during training:

```
======================================================================
                    ULTRAREFINER TRAINING
======================================================================
Experiment: BUSI_fold0
Dataset:    BUSI
Fold:       1/5
Samples:    Train: 409 | Val: 103
======================================================================

──────────────────────────────────────────────────────────────────────
  EPOCH 1/150
──────────────────────────────────────────────────────────────────────

  [TRAIN]
  ├── Loss:      0.4523
  ├── Dice:      0.7234
  ├── IoU:       0.5812
  └── LR:        9.94e-03

  [VALIDATION] ★ BEST
  ┌────────────────────────────────────────
  │  Metric              Value
  ├────────────────────────────────────────
  │  Dice                0.7512
  │  IoU (Jaccard)       0.6023
  │  Precision           0.7834
  │  Recall              0.7215
  │  Accuracy            0.9234
  │  Loss                0.4012
  └────────────────────────────────────────
```

## Pretrained Weights

1. **TransUNet ViT weights**:
   ```bash
   wget https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz -P ./pretrained/
   ```

2. **MedSAM checkpoint**: Download from [MedSAM GitHub](https://github.com/bowang-lab/MedSAM)

## Requirements

- Python >= 3.8
- PyTorch >= 1.10 (tested up to 2.6)
- CUDA >= 11.0 (recommended)

```bash
pip install -r requirements.txt
```

## Troubleshooting

### PyTorch 2.6+ Checkpoint Loading

PyTorch 2.6 changed the default value of `weights_only` from `False` to `True` in `torch.load()`. This project handles this automatically, but if you encounter errors like:

```
_pickle.UnpicklingError: Weights only load failed...
```

Ensure you're using the latest version of the scripts, which include `weights_only=False` for all checkpoint loading.

### Image Size Mismatch During Training

If you see errors about tensor size mismatch (e.g., `[224, 224]` vs `[224, 225]`), this was caused by floating-point precision issues in image resizing. The latest version uses `cv2.resize()` which produces exact dimensions.

## Citation

```bibtex
@article{ultrarefiner2024,
  title={UltraRefiner: End-to-End Differentiable Segmentation Refinement for Medical Image Analysis},
  author={},
  journal={},
  year={2024}
}
```

## References

- [TransUNet](https://arxiv.org/abs/2102.04306): Transformers Make Strong Encoders for Medical Image Segmentation
- [MedSAM](https://github.com/bowang-lab/MedSAM): Segment Anything in Medical Images
- [SAM](https://segment-anything.com/): Segment Anything Model

## License

This project is for research purposes only.
