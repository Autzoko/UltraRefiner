# UltraRefiner: Three-Phase Training Pipeline Analysis Report

## Table of Contents
1. [Project Goal](#1-project-goal)
2. [Architecture Overview](#2-architecture-overview)
3. [Phase 1: TransUNet Coarse Segmentation](#3-phase-1-transunet-coarse-segmentation)
4. [Phase 2: SAM Finetuning (Distribution Alignment)](#4-phase-2-sam-finetuning)
5. [Phase 3: End-to-End Joint Training](#5-phase-3-end-to-end-joint-training)
6. [Differentiable Pipeline Analysis](#6-differentiable-pipeline-analysis)
7. [Correctness Analysis](#7-correctness-analysis)
8. [Potential Issues and Concerns](#8-potential-issues-and-concerns)
9. [Recommendations](#9-recommendations)

---

## 1. Project Goal

Train **TransUNet** on breast ultrasound datasets, finetune **SAM** on augmented/predicted data, then jointly train both models end-to-end so SAM refines TransUNet's coarse predictions. The final goal is to improve segmentation quality beyond what TransUNet alone can achieve while keeping TransUNet's predictions stable or improving them via gradient flow from SAM.

---

## 2. Architecture Overview

```
Phase 1:  Image ──> TransUNet ──> Coarse Mask (224x224)
                                      │
Phase 2:  Augmented Masks + GT ──> SAM Decoder ──> Refined Mask (1024x1024)
                                      │
Phase 3:  Image ──> TransUNet ──> Coarse ──> [Differentiable Prompts] ──> SAM ──> Refined
                 ◄───────────── gradient flow ◄────────────────────────────────┘
```

**Key files:**
- `scripts/train_transunet.py` — Phase 1
- `scripts/finetune_sam_offline.py` — Phase 2
- `scripts/train_e2e.py` — Phase 3
- `models/ultra_refiner.py` — UltraRefiner end-to-end model
- `models/sam_refiner.py` — Differentiable SAM refinement module
- `data/mask_augmentation.py` — 12-type mask error augmentation
- `utils/losses.py` — All loss functions

---

## 3. Phase 1: TransUNet Coarse Segmentation

### 3.1 Method
- **Model:** R50-ViT-B_16 (ResNet50 stem + 12-layer ViT-B + CNN decoder with skip connections)
- **Input:** 224x224 grayscale images (1-channel), output: 2-class logits (background + foreground)
- **Loss:** `0.5 * CrossEntropy + 0.5 * Dice`
- **Optimizer:** SGD with momentum 0.9, weight decay 1e-4
- **LR Schedule:** Polynomial decay: `lr = base_lr * (1 - iter/max_iter)^0.9`
- **K-fold CV:** 5-fold within train split, best model saved per fold

### 3.2 Logic Flow
```
train_transunet.py:
  1. Load dataset (single or combined) with K-fold split
  2. Build TransUNet with optional ImageNet21K pretrained ViT weights (.npz)
  3. Train with CE+Dice loss, poly LR schedule
  4. Save best checkpoint based on validation Dice
  → Output: checkpoints/transunet/{dataset}/fold_{i}/best.pth
```

### 3.3 Correctness Assessment

**Correct:**
- Standard TransUNet training procedure, well-established
- Poly LR schedule is appropriate for medical image segmentation
- CE + Dice loss combination is standard and effective
- K-fold CV provides robust evaluation

**Potential Issues:**
1. **LR schedule timing (Minor):** The poly schedule uses `iter_num = epoch * len(dataloader)` at the start of each epoch, which means the LR is updated *after* the forward pass of each batch but *before* the gradient step. The LR is applied to the *next* step. This is technically correct but slightly unusual — most implementations update LR either before or after an entire epoch.

2. **Optimizer choice:** SGD with lr=0.01 is used (matching original TransUNet paper), which is correct. No issue here.

3. **No test-time augmentation or ensemble:** Only single-model evaluation. Not a bug, but worth noting.

---

## 4. Phase 2: SAM Finetuning

### 4.1 Method

Phase 2 has two sub-steps:

#### Step A: Generate Augmented Masks (`generate_augmented_masks.py`)
- Takes ground truth masks and applies 12 types of corruption to simulate TransUNet errors
- Each GT mask gets ~5 augmented versions
- Augmentation types include: over/under-segmentation, missing chunks, holes, false positive islands, fragmentation, shift, empty predictions, noise scatter
- Optionally generates **soft masks** via signed distance transform + sigmoid with temperature sampling T in [2, 8]

#### Step B: SAM Finetuning (`finetune_sam_offline.py`)
- **Model:** SAM ViT-B with **frozen image encoder**, trainable prompt encoder + mask decoder
- **Input:** 1024x1024 images + augmented coarse masks
- **Prompts:** Point (soft-argmax centroid), Box (center ± 2.5σ), Mask (direct [-10,+10] logits)
- **Loss:** Quality-aware loss = `BCE + Dice + λ * change_penalty`
  - Change penalty: penalizes modifying pixels where coarse mask already matches GT
- **Optimizer:** AdamW, lr=1e-4, cosine annealing
- **Optional:** Hybrid mode mixing real TransUNet predictions with augmented masks

### 4.2 Logic Flow
```
finetune_sam_offline.py:
  1. Load pre-generated augmented masks from disk
  2. Build DifferentiableSAMRefiner with frozen image encoder
  3. Train with quality-aware loss
  4. Save best SAM checkpoint based on validation refined Dice
  → Output: checkpoints/sam_finetuned/best_sam.pth + best.pth
```

### 4.3 Correctness Assessment

**Correct:**
- Freezing image encoder is standard (too expensive to finetune, and pretrained features are strong)
- Quality-aware loss prevents SAM from "correcting" already-correct regions (prevents learning identity mapping)
- Mask augmentation covers realistic failure modes
- Soft mask generation via SDF is differentiable-compatible

**Potential Issues:**

1. **`change_penalty_weight` hardcoded in training loop (Minor Bug):**
   - In `train_epoch()` at line 167, the `change_penalty_weight` is hardcoded to `0.5`:
     ```python
     loss, loss_components = quality_aware_loss(
         refined_masks, gt_mask, coarse_mask, change_penalty_weight=0.5
     )
     ```
   - The command-line argument `--change_penalty_weight` (default 0.5) is parsed but **never passed** to the loss function. It happens to match the hardcoded default, so this is not causing actual bugs, but it means the CLI argument is non-functional.

2. **SAM checkpoint saving inconsistency (Potential Issue):**
   - At line 381-382, SAM weights are saved with prefix filtering:
     ```python
     sam_state = {k: v for k, v in model.state_dict().items() if k.startswith('sam.')}
     ```
   - This saves with `sam.` prefix (e.g., `sam.image_encoder.xxx`). But when loading in Phase 3 via `sam_model_registry[type](checkpoint=path)`, SAM expects keys **without** the `sam.` prefix. The `best_sam.pth` file will fail to load directly into SAM. Only `best.pth` (the full DifferentiableSAMRefiner state) would work, but Phase 3 loads via `sam_model_registry` which expects original SAM format.
   - **This is a significant loading issue** — you need to either strip the `sam.` prefix when saving, or load the full refiner checkpoint and extract weights differently in Phase 3.

3. **Image normalization path (Potential Issue):**
   - Phase 2 dataset (`OfflineAugmentedDataset`) outputs SAM-normalized images (pixel mean/std subtracted)
   - The refiner is called with `image_already_normalized=True`
   - But `DifferentiableSAMRefiner.forward()` with `image_already_normalized=True` only pads the image, it does NOT resize it. The dataset should output images at exactly 1024x1024 already padded — this seems to be handled correctly in `SAMRandomGenerator` which does aspect-preserving resize + padding.
   - However, in Phase 3, `UltraRefiner.prepare_sam_input()` does its own resize+pad+normalize. If Phase 2 and Phase 3 normalize differently (e.g., different padding sizes due to different original aspect ratios), there could be a distribution mismatch. This is a subtle but real concern.

---

## 5. Phase 3: End-to-End Joint Training

### 5.1 Method
- **Model:** `UltraRefiner` = TransUNet + DifferentiableSAMRefiner
- **Input:** 224x224 images (same as Phase 1)
- **Pipeline:**
  1. TransUNet produces 2-class logits → softmax → foreground probability (coarse mask)
  2. Image is resized to 1024x1024 + SAM-normalized
  3. Coarse mask resized to 1024x1024
  4. Differentiable prompts extracted: point (soft-argmax), box (center ± 2.5σ), mask (direct/gaussian)
  5. SAM decoder produces 3 candidate masks + IoU predictions
  6. Soft mask selection via softmax-weighted combination
  7. Refined mask resized back to 224x224

- **Loss:** `λ_c * (0.5*CE + 0.5*Dice) + λ_r * (BCE + Dice)` where λ_c=0.5, λ_r=0.5
- **Optimizer:** AdamW with separate LRs: TransUNet lr=1e-5, SAM lr=1e-5
- **Protection mechanisms:**
  - Gradient scaling: TransUNet receives α% of SAM gradients (default α=1.0)
  - Weight regularization: L2 penalty anchoring TransUNet to Phase 1 weights
  - Optional TransUNet freezing for initial epochs
  - Gradient clipping (max norm 1.0)

### 5.2 Logic Flow
```
train_e2e.py:
  1. Build UltraRefiner with Phase 1 TransUNet + Phase 2 SAM checkpoints
  2. Evaluate TransUNet baseline before training
  3. Set up gradient scaler and weight regularizer
  4. Optionally freeze TransUNet/SAM for initial epochs
  5. Train with EndToEndLoss (coarse + refined)
  6. Monitor TransUNet performance vs baseline (warn if drops >2%)
  7. Save best checkpoint based on refined Dice
  → Output: checkpoints/ultra_refiner/fold_{i}/best.pth
```

### 5.3 Correctness Assessment

**Correct:**
- Differentiable prompt extraction (soft-argmax, weighted statistics) maintains gradient flow
- Soft mask selection avoids non-differentiable argmax
- Separate learning rates for TransUNet and SAM are important
- TransUNet protection mechanisms (grad scaling, weight reg) prevent catastrophic forgetting
- Baseline evaluation before training enables monitoring TransUNet degradation

**Potential Issues:**

1. **GradientScaler closure bug (Bug):**
   - In `train_e2e.py` line 799:
     ```python
     for param in model.transunet.parameters():
         if param.requires_grad:
             handle = param.register_hook(lambda grad: grad * scale)
     ```
   - This is a classic Python closure bug. The lambda captures `scale` by reference, which is fine here since `scale` doesn't change. **However**, this is only safe because `scale` is set once. If the code were ever modified to change `scale` dynamically, all hooks would use the latest value. Worth noting but not currently buggy.

2. **Refined mask output space inconsistency (Important):**
   - In the standard (non-gated) UltraRefiner flow:
     - `sam_output['masks']` from `DifferentiableSAMRefiner` are **logits** (not probabilities)
     - These logits are stored in `result['refined_mask']` and `result['refined_mask_logits']`
     - In `train_e2e.py` line 598: `refined_pred = torch.sigmoid(outputs['refined_mask'])`
     - But in `EndToEndLoss.forward()` line 465: `refined_loss = self.refined_loss_fn(refined_mask.unsqueeze(1), ...)` where `self.refined_loss_fn = BCEDiceLoss()` which applies `BCEWithLogitsLoss` internally
     - So: loss receives logits (correct), metrics receive sigmoid(logits) (correct). **This is actually consistent.**
   - In gated mode: `refined_mask` is **already probabilities**, and `BCEDiceLossProb` is used. Also consistent.
   - **Assessment: Correct implementation**, but the dual output naming (`refined_mask` vs `refined_mask_logits`) is confusing. In standard mode, `refined_mask` IS the logits resized to 224x224, while `refined_mask_logits` is the logits at 1024x1024.

3. **Resolution mismatch in loss computation (Important):**
   - The refined loss is computed at 224x224 (default `refined_eval_size=224`)
   - The SAM output is at 1024x1024, then resized to 224x224 via bilinear interpolation
   - This downsampling can blur sharp boundaries that SAM produces
   - The `--refined_eval_size 1024` option exists but requires upsampling the GT label with nearest interpolation, which is also imperfect
   - **Recommendation:** Consider computing loss at 1024x1024 by upsampling GT with nearest to preserve SAM's boundary detail

4. **SAM image encoder is run with `torch.set_grad_enabled(self.sam.image_encoder.training)` (Correct but subtle):**
   - Since the image encoder is frozen (`training=False`), gradients are disabled for the encoder forward pass
   - This is memory-efficient but means no gradients flow through the image encoder even if someone accidentally unfreezes it
   - This is actually the intended behavior

5. **Coarse mask resizing for SAM (Important Potential Issue):**
   - `coarse_mask_resized = F.interpolate(..., size=(1024, 1024), mode='bilinear')`
   - The coarse mask is a soft probability, resized from 224 to 1024 with bilinear interpolation
   - In Phase 2, SAM was trained on augmented masks that may have different characteristics (e.g., soft SDF masks with temperatures, or binary masks with morphological corruption)
   - If Phase 2 used `--mask_prompt_style direct`, the mask prompt conversion is: `logits = (prob * 2 - 1) * 10`, mapping [0,1] to [-10,10]
   - The bilinear upscaling of the coarse mask from 224→1024 produces smooth transitions, which should match Phase 2's soft mask distribution reasonably well
   - **If Phase 2 used binary (hard) masks** but Phase 3 feeds soft masks, there's a distribution mismatch

6. **Optimizer param groups mismatch on unfreeze (Potential Issue):**
   - When TransUNet is frozen initially and later unfrozen, new params are added via `optimizer.add_param_group()`
   - The `CosineAnnealingLR` scheduler was created with the *original* optimizer state (without TransUNet params)
   - After adding a new param group, the scheduler's internal state may not properly handle the new group
   - This could result in the TransUNet learning rate not following the cosine schedule correctly
   - **This is a known PyTorch issue** with `add_param_group` + schedulers

---

## 6. Differentiable Pipeline Analysis

### 6.1 Gradient Flow Path
```
Loss_refined
  └──> refined_mask (224x224 logits via bilinear resize from 1024x1024)
         └──> soft_mask_selection (softmax-weighted combination of 3 masks)
                └──> SAM mask_decoder (cross-attention transformer)
                       └──> prompt_encoder
                              ├──> sparse_embeddings (from points + boxes)
                              │       ├──> point_coords ← soft-argmax(coarse_mask)  ✓ differentiable
                              │       └──> box_coords ← center ± 2.5σ(coarse_mask)  ✓ differentiable
                              └──> dense_embeddings (from mask input)
                                      └──> mask_input ← (coarse_mask * 2 - 1) * 10  ✓ differentiable

Loss_coarse
  └──> transunet_output ──> TransUNet ──> image  (standard backprop)
```

### 6.2 Critical Differentiability Points

| Component | Method | Differentiable? | Notes |
|-----------|--------|----------------|-------|
| Point extraction | Soft-argmax (weighted centroid) | Yes | Gradient = (coord - center) / sum |
| Box extraction | Center ± 2.5 * weighted_std | Yes | Via weighted variance |
| Negative point | Soft box mask + weighted centroid on (1 - mask) | Yes | Sigmoid soft boundaries |
| Mask prompt | Linear scaling to [-10, 10] or Gaussian blur | Yes | All ops are differentiable |
| Mask selection | Softmax(IoU/τ) weighted combination | Yes | Temperature τ=0.1 |
| ROI cropping | F.grid_sample with affine_grid | Yes | Fully differentiable |
| ROI paste back | Inverse F.grid_sample with soft ROI mask | Yes | Soft blending |

**Assessment:** The differentiable pipeline is well-designed. All operations maintain gradient flow from the refined loss back to TransUNet's parameters. The `torch.clamp` operations in box extraction can cause zero gradients at boundaries, but this is unlikely to be an issue in practice since masks are soft probabilities.

---

## 7. Correctness Analysis

### 7.1 Confirmed Correct Implementations

1. **TransUNet architecture and training** — Standard implementation matching the original paper
2. **SAM integration** — Correct use of SAM's prompt encoder and mask decoder
3. **Differentiable prompt extraction** — Mathematically sound soft-argmax and weighted statistics
4. **Quality-aware loss in Phase 2** — Properly prevents over-correction of correct regions
5. **K-fold cross-validation** — Correct implementation with deterministic splits (seeded)
6. **Mask augmentation** — Comprehensive 12-type error simulation with appropriate probabilities
7. **Gated residual refinement** — Well-designed uncertainty-based gating mechanism
8. **Weight regularization** — Correct L2 penalty anchoring to Phase 1 weights

### 7.2 Minor Issues (Functional but Suboptimal)

1. **Hardcoded `change_penalty_weight=0.5`** in `finetune_sam_offline.py` — CLI arg is ignored
2. **Confusing output naming** — `refined_mask` contains logits in standard mode but probabilities in gated mode
3. **No learning rate warmup in Phase 3** — CosineAnnealingLR starts at peak LR immediately, which can destabilize early training

### 7.3 Significant Issues

1. **SAM checkpoint format mismatch** — `best_sam.pth` saved with `sam.` prefix won't load via `sam_model_registry`
2. **Scheduler + add_param_group interaction** — CosineAnnealingLR may not handle dynamically added param groups
3. **Resolution mismatch in loss** — Computing refined loss at 224x224 discards SAM's boundary improvements
4. **Phase 2/3 distribution gap** — Phase 2 trains on augmented GT masks at 1024x1024; Phase 3 feeds bilinear-upscaled soft TransUNet output at 1024x1024. The distributions may differ significantly.

---

## 8. Potential Issues and Concerns

### 8.1 Data Flow Concerns

| Stage | Phase 2 | Phase 3 | Match? |
|-------|---------|---------|--------|
| Image size | 1024x1024 (native SAM) | 224x224 → upsample to 1024x1024 | Partial |
| Image channels | 3-channel (replicated grayscale) | 1-channel → replicate to 3 | Yes |
| Image normalization | SAM pixel mean/std | SAM pixel mean/std | Yes |
| Coarse mask source | Augmented GT (12 error types) | TransUNet softmax output | Different distribution |
| Coarse mask values | Binary or soft (SDF) | Continuous probability [0,1] | Different distribution |
| Mask prompt conversion | `(mask * 2 - 1) * 10` | `(mask * 2 - 1) * 10` | Yes (if both use `direct`) |
| Coordinate space | 1024x1024 native | 1024x1024 (after upscale) | Yes |

**Key concern:** The coarse masks in Phase 2 are augmented GT masks (structurally corrupted ground truth), while in Phase 3 they are real TransUNet outputs (soft probabilities with CNN-like artifacts). SAM may behave differently on these two distributions. **Using hybrid training (mixing real predictions with augmented masks) partially addresses this.**

### 8.2 Memory Concerns

- Phase 3 runs both TransUNet (224x224) and SAM (1024x1024) in a single forward pass
- With batch_size=8, SAM image encoder alone needs ~4GB VRAM (ViT-B)
- SAM image encoder is frozen but still requires memory for forward pass
- Gradient accumulation is available but defaults to 1
- **Recommendation:** Monitor GPU memory; batch_size=4 may be needed on GPUs with <24GB

### 8.3 Training Stability Concerns

- TransUNet can degrade during E2E training if SAM gradients are too strong
- The `--transunet_grad_scale` (default 1.0) means full gradients flow to TransUNet
- **Recommendation:** Use `--transunet_grad_scale 0.1` and `--transunet_weight_reg 0.01` as documented

---

## 9. Recommendations

### 9.1 Critical Fixes

1. **Fix SAM checkpoint saving in Phase 2:**
   ```python
   # Current (broken for Phase 3 loading):
   sam_state = {k: v for k, v in model.state_dict().items() if k.startswith('sam.')}

   # Fix: Strip 'sam.' prefix OR strip 'sam.sam.' prefix depending on nesting
   sam_state = {k.replace('sam.', '', 1): v for k, v in model.state_dict().items() if k.startswith('sam.')}
   ```
   Alternatively, in Phase 3's `build_ultra_refiner`, load the SAM checkpoint manually after construction instead of via `sam_model_registry`.

2. **Pass `change_penalty_weight` from args** in `finetune_sam_offline.py`:
   ```python
   loss, loss_components = quality_aware_loss(
       refined_masks, gt_mask, coarse_mask, change_penalty_weight=args.change_penalty_weight
   )
   ```

### 9.2 Important Improvements

3. **Add learning rate warmup in Phase 3:** The CosineAnnealingLR starts at peak immediately. Use `CosineAnnealingWarmRestarts` or a linear warmup for the first 5-10 epochs to prevent early destabilization.

4. **Use `--transunet_grad_scale 0.1`** as default instead of 1.0 to better protect TransUNet from degradation.

5. **Consider computing refined loss at 1024x1024** by upsampling GT labels with nearest interpolation, preserving SAM's boundary improvements.

6. **Increase hybrid training ratio:** When Phase 2 is followed by Phase 3, using some real TransUNet predictions (`--real_ratio 0.3`) in Phase 2 helps bridge the distribution gap.

### 9.3 Best Practice Training Pipeline

```bash
# Phase 1: Train TransUNet (per dataset or combined)
python scripts/train_transunet.py \
    --data_root ./dataset/processed \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --fold 0 --max_epochs 150 --batch_size 24 --base_lr 0.01 \
    --vit_pretrained ./pretrained/R50+ViT-B_16.npz

# Phase 2a: Generate augmented masks
python scripts/generate_augmented_masks.py \
    --data_root ./dataset/processed \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --output_dir ./dataset/augmented_masks \
    --num_augmentations 5 --soft_mask_prob 0.5

# Phase 2b: Finetune SAM
python scripts/finetune_sam_offline.py \
    --data_root ./dataset/augmented_masks \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --sam_checkpoint ./pretrained/sam_vit_b_01ec64.pth \
    --epochs 100 --batch_size 4 --lr 1e-4 \
    --mask_prompt_style direct --use_roi_crop --use_amp

# Phase 3: End-to-end training
python scripts/train_e2e.py \
    --data_root ./dataset/processed \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --transunet_checkpoint ./checkpoints/transunet/combined/fold_0/best.pth \
    --sam_checkpoint ./checkpoints/sam_finetuned_offline/best_sam.pth \
    --fold 0 --max_epochs 100 --batch_size 8 \
    --transunet_lr 1e-5 --sam_lr 1e-5 \
    --mask_prompt_style direct --use_roi_crop \
    --transunet_grad_scale 0.1 --transunet_weight_reg 0.01 \
    --coarse_loss_weight 0.5 --refined_loss_weight 0.5
```

### 9.4 Summary Table

| Aspect | Status | Notes |
|--------|--------|-------|
| Phase 1 (TransUNet) | Correct | Standard implementation |
| Phase 2 (SAM finetune) | Mostly correct | SAM checkpoint format issue |
| Phase 3 (E2E) | Mostly correct | Scheduler/unfreeze interaction, resolution mismatch |
| Differentiable pipeline | Correct | Well-designed gradient flow |
| Loss functions | Correct | Appropriate for each phase |
| Data augmentation | Correct | Comprehensive mask corruption |
| K-fold CV | Correct | Deterministic with seed |
| Gated refinement | Correct | Alternative for unfinetuned SAM |
| Memory efficiency | Good | Frozen encoder, AMP support |
| Training stability | Needs attention | Default grad_scale=1.0 too aggressive |

---

*Report generated for UltraRefiner project analysis. Last updated: Jan 2026.*
