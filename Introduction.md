# UltraRefiner: End-to-End Differentiable Segmentation Refinement with Gradient-Coupled Foundation Models

## 1. Introduction and Motivation

### 1.1 The Challenge of Medical Image Segmentation

Medical image segmentation represents one of the most fundamental and challenging tasks in computer-aided diagnosis. In breast ultrasound imaging specifically, the task is complicated by several inherent characteristics of the imaging modality:

- **Low Contrast**: Breast ultrasound images exhibit poor contrast between lesion tissue and surrounding parenchyma, making boundary delineation inherently ambiguous.
- **Speckle Noise**: The coherent nature of ultrasound produces multiplicative speckle noise that obscures fine structural details.
- **Ill-Defined Boundaries**: Tumor margins are often infiltrative rather than well-circumscribed, leading to gradual intensity transitions rather than sharp edges.
- **Heterogeneous Morphology**: Breast lesions exhibit enormous variability in shape, size, echogenicity, and internal texture, making generalization difficult.
- **Acoustic Shadowing and Enhancement**: Posterior acoustic artifacts can obscure or mimic lesion boundaries.

These challenges result in segmentation outputs that frequently suffer from imprecise boundaries, topological errors (holes, bridges, fragmentation), and catastrophic failures on ambiguous cases.

### 1.2 The Promise and Limitation of Foundation Models

The Segment Anything Model (SAM), trained on over 1 billion masks across 11 million images, represents a paradigm shift in image segmentation. SAM's prompt-based architecture enables powerful zero-shot segmentation capabilities through a flexible prompting mechanism that accepts points, bounding boxes, and coarse masks as input. This architecture makes SAM an ideal candidate for refining coarse segmentation predictions from task-specific networks.

However, naive integration of SAM into a medical image segmentation pipeline faces two fundamental obstacles:

1. **Prompt Extraction Non-Differentiability**: SAM requires carefully designed prompts (points, boxes, masks) that must be extracted from upstream predictions. Standard extraction methods—argmax for point selection, thresholding for binarization—are non-differentiable operations that block gradient flow.

2. **Distribution Mismatch**: SAM was trained on binary masks with sharp, well-defined boundaries. In contrast, learned segmentation networks produce soft probability maps with uncertain, blurred boundaries. This distribution gap causes SAM to underperform when receiving soft masks as input.

These limitations prevent end-to-end optimization, forcing practitioners to use SAM only as a post-processing step that cannot influence the upstream network's learning.

### 1.3 UltraRefiner: A Unified Solution

UltraRefiner addresses these fundamental limitations by introducing a **fully differentiable two-stage segmentation refinement framework** that enables gradient flow from the final refined output through SAM's prompt encoding back to the coarse segmentation network. This end-to-end differentiability allows joint optimization where:

- **TransUNet** (the coarse network) learns to produce outputs that are optimally suited for SAM refinement
- **SAM** (the refiner) learns to correct the specific failure modes of the upstream network

The result is a synergistic system where both networks co-adapt during training, achieving performance superior to either network alone or their naive cascade.

---

## 2. Model Architecture

### 2.1 TransUNet: The Coarse Segmentation Network

TransUNet serves as the coarse segmentation backbone, combining the locality of Convolutional Neural Networks with the global attention mechanism of Vision Transformers.

#### 2.1.1 Encoder Architecture

The encoder follows a hybrid CNN-Transformer design:

1. **CNN Feature Extraction**: A pretrained ResNet-50 extracts hierarchical features at multiple scales. The network processes 224×224 input images through four residual stages, producing feature maps at 1/4, 1/8, 1/16, and 1/32 of the original resolution.

2. **Patch Embedding**: The final CNN feature map (14×14 for 224 input) is flattened into a sequence of 196 patch tokens, each of dimension 768.

3. **Transformer Encoding**: A 12-layer Vision Transformer (ViT-B/16) processes the patch sequence. Each layer consists of:
   - Multi-Head Self-Attention (MHSA) with 12 attention heads
   - Layer Normalization before each sub-layer
   - MLP with GELU activation and hidden dimension 3072
   - Residual connections around each sub-layer

The Transformer captures long-range dependencies that are critical for understanding global context in medical images—for instance, recognizing that a shadow artifact below a lesion is not part of the lesion itself.

#### 2.1.2 Decoder Architecture

The decoder progressively upsamples features back to the original resolution:

1. **Reshape**: Transformer output is reshaped from sequence (196×768) back to spatial format (14×14×768)

2. **Progressive Upsampling**: Four upsampling stages, each consisting of:
   - 2× bilinear upsampling
   - Concatenation with skip connection from corresponding ResNet stage
   - Two 3×3 convolutional layers with batch normalization and ReLU

3. **Output Head**: Final 1×1 convolution produces 2-channel logits (background, foreground)

#### 2.1.3 Output Processing

The network outputs logits that are converted to soft probability masks:

```
P(lesion) = softmax(logits)[:, 1]  ∈ [0, 1]
```

This soft probability mask preserves uncertainty information that is crucial for downstream refinement—regions with P ≈ 0.5 indicate boundary uncertainty where SAM's refinement is most valuable.

### 2.2 Segment Anything Model (SAM): The Refinement Network

SAM consists of three components that work together to produce refined segmentation masks from various prompt types.

#### 2.2.1 Image Encoder

SAM's image encoder is a Vision Transformer pretrained using Masked Autoencoder (MAE) on a massive image dataset:

- **Architecture**: ViT-B/16 (base), ViT-L/16 (large), or ViT-H/16 (huge)
- **Input Resolution**: 1024×1024 pixels
- **Output**: Dense feature map of shape 64×64×256

The image encoder is computationally expensive but can be computed once and cached, as it depends only on the image and not on the prompts.

#### 2.2.2 Prompt Encoder

The prompt encoder converts various prompt types into embeddings that condition mask generation:

**Point Prompts**: Each point is encoded as:
- Positional encoding using Fourier features
- Learned embedding indicating foreground (+) or background (−) label
- Points are processed as sparse prompts

**Box Prompts**: Bounding boxes are encoded as:
- Two corner points (top-left, bottom-right) with positional encoding
- Learned embeddings distinguishing corners
- Processed as sparse prompts

**Mask Prompts**: Coarse masks are encoded as:
- Downsampled to 256×256
- Processed through lightweight convolutional layers
- Added to dense image embeddings
- Processed as dense prompts

#### 2.2.3 Mask Decoder

The mask decoder is a modified Transformer decoder that generates segmentation masks:

1. **Cross-Attention**: Output tokens attend to image embeddings (combined with prompt embeddings)

2. **Self-Attention**: Output tokens attend to each other and to prompt tokens

3. **MLP Heads**:
   - Mask prediction head: Produces 3 candidate masks at 256×256 resolution
   - IoU prediction head: Predicts quality score for each mask

4. **Upsampling**: Masks are upsampled to original resolution using transposed convolutions

The multi-mask output with IoU scoring enables the model to express ambiguity and select the most appropriate segmentation.

### 2.3 UltraRefiner: The Unified Architecture

UltraRefiner integrates TransUNet and SAM into a unified, end-to-end differentiable architecture.

#### 2.3.1 Forward Pass

1. **Coarse Segmentation**: Input image passes through TransUNet to produce soft probability mask P(x) ∈ [0,1]

2. **Resolution Alignment**: Coarse mask (224×224) is upsampled to SAM's resolution (1024×1024)

3. **Differentiable Prompt Extraction**: Points, box, and mask prompts are extracted from the soft mask using fully differentiable operations (detailed in Section 3)

4. **SAM Refinement**: Prompts condition SAM's mask decoder to produce refined segmentation

5. **Multi-Mask Selection**: Three candidate masks are combined using soft IoU-weighted averaging

6. **Output**: Final refined mask at 1024×1024 or downsampled to match label resolution

#### 2.3.2 Gradient Flow

The key innovation is maintaining differentiability throughout the pipeline:

```
∂L/∂θ_TransUNet = ∂L/∂mask_refined × ∂mask_refined/∂prompts × ∂prompts/∂mask_coarse × ∂mask_coarse/∂θ_TransUNet
```

This enables TransUNet to receive gradients that encode "how should I change my prediction to make SAM produce better refinements?"

---

## 3. Differentiable Prompt Extraction

The central technical contribution of UltraRefiner is a suite of fully differentiable prompt extraction mechanisms that operate on soft probability masks rather than binary predictions.

### 3.1 Point Prompts via Soft-Argmax

Traditional point extraction uses argmax to find the maximum probability location—a non-differentiable operation. UltraRefiner replaces this with probability-weighted spatial expectation.

#### 3.1.1 Foreground Point Extraction

The foreground centroid is computed as:

```
x_fg = Σᵢⱼ P(i,j) × xⱼ / Σᵢⱼ P(i,j)
y_fg = Σᵢⱼ P(i,j) × yᵢ / Σᵢⱼ P(i,j)
```

where P(i,j) is the soft mask probability at spatial location (i,j), and (xⱼ, yᵢ) are the coordinate values.

**Gradient Analysis**: The gradient of the centroid with respect to the mask is:

```
∂x_fg/∂P(i,j) = (xⱼ - x_fg) / Σ P
```

This gradient has an intuitive interpretation: pixels far from the current centroid receive larger gradient magnitudes, encouraging the mask to "pull" the centroid toward better locations.

#### 3.1.2 Background Point Extraction

Background points should lie outside the lesion but within a contextually relevant region. UltraRefiner extracts background points from the inverse mask within the predicted bounding box:

```
P_bg(i,j) = (1 - P(i,j)) × I_box(i,j)
```

where I_box is an indicator function for the bounding box region. The background centroid is then computed analogously to the foreground centroid.

#### 3.1.3 Temperature Scaling

Optional temperature scaling sharpens the probability distribution before expectation:

```
P_sharp(i,j) = P(i,j)^(1/τ) / Σ P(i,j)^(1/τ)
```

Lower temperature τ concentrates mass on high-probability regions, approximating argmax while remaining differentiable.

### 3.2 Box Prompts via Weighted Statistics

Bounding box extraction traditionally uses thresholding followed by min/max coordinate extraction. UltraRefiner replaces this with mask-weighted coordinate statistics.

#### 3.2.1 Weighted Mean and Standard Deviation

```
μ_x = Σᵢⱼ P(i,j) × xⱼ / Σᵢⱼ P(i,j)
σ_x = √(Σᵢⱼ P(i,j) × (xⱼ - μ_x)² / Σᵢⱼ P(i,j))
```

(analogously for y-coordinates)

#### 3.2.2 Box Construction

The bounding box is constructed as:

```
x₁ = μ_x - k × σ_x
x₂ = μ_x + k × σ_x
y₁ = μ_y - k × σ_y
y₂ = μ_y + k × σ_y
```

where k = 2.5 ensures approximately 99% coverage of a Gaussian distribution.

#### 3.2.3 Advantages over Soft Min/Max

An alternative approach would use soft-min/max:

```
x_min = Σ xⱼ × exp(-xⱼ/τ) / Σ exp(-xⱼ/τ)
```

However, this approach is dominated by coordinate values regardless of mask probability—a pixel at x=0 with P=0.001 contributes as much as a pixel at x=0 with P=0.999. The weighted statistics approach correctly restricts computation to high-probability regions.

### 3.3 Mask Prompts via Adaptive Conversion

SAM expects mask prompts as logits (unbounded real values) rather than probabilities. UltraRefiner provides two conversion strategies.

#### 3.3.1 Direct Conversion

For SAM models finetuned on soft masks (Phase 2 trained):

```
logits = (P × 2 - 1) × scale
```

This maps [0,1] → [-scale, +scale]. With scale=10, confident predictions (P≈0 or P≈1) map to logits of magnitude 10, while uncertain predictions (P≈0.5) map to logits near 0.

#### 3.3.2 Gaussian Conversion

For SAM models not finetuned on soft masks (e.g., pretrained SAM, MedSAM):

```
P_smoothed = GaussianBlur(P, σ=adaptive)
logits = (P_smoothed × 2 - 1) × scale
```

The Gaussian blur creates softer boundaries that better match SAM's training distribution of sharp binary masks, preventing artifacts from abrupt probability transitions.

#### 3.3.3 Distance Transform Conversion

An alternative approach uses signed distance transform:

```
dist_inside = DistanceTransform(P > 0.5)
dist_outside = DistanceTransform(P ≤ 0.5)
signed_dist = dist_inside - dist_outside
logits = signed_dist / temperature
```

This creates smooth transitions at boundaries with magnitude proportional to distance from the boundary.

---

## 4. Three-Phase Curriculum Training Strategy

UltraRefiner employs a carefully designed three-phase training curriculum that progressively builds the joint system from independent components to full end-to-end optimization.

### 4.1 Phase 1: Coarse Network Training

#### 4.1.1 Objective

Train TransUNet independently to establish a strong baseline coarse segmentation model.

#### 4.1.2 Training Protocol

- **Loss Function**: Combined Cross-Entropy and Dice loss
  ```
  L = 0.5 × CE(logits, GT) + 0.5 × Dice(softmax(logits), GT)
  ```

- **Data Augmentation**: Random rotation, flipping, scaling, elastic deformation

- **Cross-Validation**: K-fold (typically K=5) cross-validation to:
  - Evaluate generalization performance
  - Generate out-of-fold predictions for Phase 2
  - Enable ensemble methods

#### 4.1.3 Outcome

Phase 1 produces:
- Trained TransUNet checkpoints for each fold
- Characteristic failure patterns that Phase 2 must address
- Out-of-fold predictions representing realistic coarse mask quality

### 4.2 Phase 2: SAM Distribution Alignment

Phase 2 is the critical bridge between independent training and end-to-end optimization.

#### 4.2.1 The Distribution Gap Problem

TransUNet produces soft probability masks with characteristics different from SAM's training distribution:
- Uncertain boundaries (P ≈ 0.5) rather than sharp transitions
- Smooth probability gradients rather than binary values
- Various failure modes (holes, over-segmentation, etc.) not seen in SAM's training

Without alignment, SAM interprets uncertain regions incorrectly, leading to degraded refinement.

#### 4.2.2 Mask Augmentation System

To bridge the distribution gap, UltraRefiner employs a comprehensive mask augmentation system with **12 primary error types** that simulate realistic TransUNet failure patterns:

| Error Type | Simulation Method | Purpose |
|------------|-------------------|---------|
| **Identity** | Minor perturbation (0-3px) | Preserve already-good predictions |
| **Over-Segmentation** | Morphological dilation (1.2-3×) | Teach boundary contraction |
| **Giant Over-Segmentation** | Extreme dilation (3-20×) | Handle severe over-prediction |
| **Under-Segmentation** | Morphological erosion (0.4-0.9×) | Teach boundary expansion |
| **Missing Chunk** | Wedge/blob cutout (5-30%) | Teach region completion |
| **Internal Holes** | Random interior voids (2-20% each) | Teach hole filling |
| **Bridge/Adhesion** | Thin connecting bands | Teach artifact removal |
| **False Positive Islands** | Scattered spurious blobs (1-30) | Teach false positive suppression |
| **Fragmentation** | Cuts through mask (1-5) | Teach fragment merging |
| **Spatial Shift** | Translation (5-30%) | Teach location correction |
| **Empty Prediction** | Complete miss | Recovery from catastrophic failure |
| **Noise Scatter** | Random noise pattern | Handle pure noise input |

#### 4.2.3 Soft Mask Conversion

After geometric augmentation, masks undergo soft conversion to match TransUNet's output characteristics:

```
signed_dist(x) = dist_inside(x) - dist_outside(x)
P(x) = σ(signed_dist(x) / temperature)
```

where temperature ∈ [2.0, 8.0] is randomly sampled to create varying degrees of boundary blur.

#### 4.2.4 Quality-Aware Loss Function

A novel loss function prevents SAM from "correcting" regions that the coarse mask already predicted correctly:

```
L = L_BCE + L_Dice + λ × L_change_penalty
```

where:
```
L_change_penalty = mean(|σ(SAM) - coarse|² × I(coarse_binary == GT))
```

The indicator function I(·) identifies pixels where the coarse mask (after binarization) matches the ground truth. The penalty discourages changes in these regions.

**Intuition**: If the coarse mask already correctly predicts a region, SAM should not change it. This teaches SAM to be conservative—making corrections only where confident improvement is possible.

#### 4.2.5 Hybrid Training

Optionally combine augmented masks with real TransUNet predictions:

```
training_batch = α × real_predictions + (1-α) × augmented_masks
```

where α ∈ [0, 1] controls the mixing ratio. Hybrid training captures actual failure modes while augmentation provides diversity and coverage of edge cases.

### 4.3 Phase 3: End-to-End Joint Optimization

#### 4.3.1 Joint Training Objective

Both networks are trained jointly with a combined loss:

```
L = λ_coarse × L_coarse(TransUNet, GT) + λ_refined × L_refined(SAM, GT) + λ_reg × L_regularization
```

where:
- L_coarse maintains TransUNet's standalone quality
- L_refined optimizes the final refinement quality
- L_regularization prevents TransUNet from drifting too far from Phase 1

#### 4.3.2 TransUNet Protection Mechanisms

End-to-end training risks degrading TransUNet's performance as it co-adapts with SAM. Several mechanisms prevent this:

**Gradient Scaling**: Scale gradients flowing to TransUNet by factor α ∈ [0.01, 0.1]:
```
∂L/∂θ_TransUNet ← α × ∂L/∂θ_TransUNet
```
This ensures SAM adapts faster than TransUNet, preventing TransUNet from making drastic changes.

**Weight Regularization**: L2 penalty anchoring weights to Phase 1 values:
```
L_reg = β × ||θ - θ_phase1||²
```
This elastic constraint allows adaptation while preventing catastrophic forgetting.

**Two-Stage Unfreezing**: Optionally freeze TransUNet for initial epochs:
- Epochs 1-N: TransUNet frozen, only SAM trains
- Epochs N+1 onwards: Both networks train jointly

This allows SAM to first adapt to TransUNet's current behavior before joint optimization.

**Dual Loss Supervision**: Maintain explicit supervision on TransUNet's output:
```
λ_coarse ≥ λ_refined (typically 0.5-0.8 vs 0.2-0.5)
```
Higher weight on coarse loss prevents TransUNet from learning to produce outputs that are "easy for SAM" but poor standalone predictions.

#### 4.3.3 Learning Rate Strategy

Separate learning rates for each component:
- TransUNet: Lower learning rate (1e-5 to 1e-6)
- SAM Prompt Encoder: Moderate learning rate (1e-5)
- SAM Mask Decoder: Moderate learning rate (1e-5)
- SAM Image Encoder: Typically frozen (learning rate = 0)

---

## 5. Differentiable ROI Cropping

### 5.1 Motivation

Breast lesions typically occupy a small fraction of the ultrasound image. Processing the entire image at SAM's native resolution (1024×1024) wastes computation on background regions and limits effective resolution for the lesion itself.

### 5.2 ROI Extraction Pipeline

#### 5.2.1 Soft Bounding Box Extraction

The ROI bounding box is extracted from the coarse mask using the weighted statistics method (Section 3.2):

```
box = [μ_x - 2.5σ_x, μ_y - 2.5σ_y, μ_x + 2.5σ_x, μ_y + 2.5σ_y]
```

#### 5.2.2 Context Expansion

The box is expanded by a configurable ratio (default 30%) to include surrounding context:

```
box_expanded = box + expand_ratio × (box_size)
```

Context is crucial for SAM to understand the local image structure and produce accurate boundaries.

#### 5.2.3 Differentiable Cropping

Cropping is performed using grid_sample, which is fully differentiable:

```
grid = normalize(box_coordinates)
cropped = F.grid_sample(image, grid, mode='bilinear', align_corners=False)
```

Bilinear interpolation ensures smooth gradients flow through the crop operation.

#### 5.2.4 Resolution Upscaling

The cropped ROI is resized to 1024×1024 (SAM's native resolution):

```
roi_1024 = F.interpolate(cropped, size=(1024, 1024), mode='bilinear')
```

For a lesion occupying 100×100 pixels in the original image, this provides 10× resolution enhancement.

#### 5.2.5 SAM Processing

SAM processes the high-resolution ROI to produce refined segmentation within the cropped region.

#### 5.2.6 Differentiable Paste-Back

The refined mask is placed back into the original image coordinates using inverse grid_sample:

```
full_mask = F.grid_sample(roi_mask, inverse_grid, mode='bilinear')
```

Regions outside the ROI retain the coarse mask prediction (or zeros).

### 5.3 Benefits

1. **Resolution Enhancement**: 4-16× effective resolution for small lesions
2. **Computational Efficiency**: Reduces unnecessary computation on background
3. **Gradient Flow**: Fully differentiable—gradients flow through crop coordinates back to coarse mask
4. **Context Preservation**: Expansion ratio ensures sufficient surrounding context

---

## 6. Gated Residual Refinement

### 6.1 Motivation

In scenarios where Phase 2 finetuning is impractical (e.g., using pretrained SAM/MedSAM without modification), unrestricted SAM refinement may degrade accurate coarse predictions. Gated residual refinement constrains SAM to act as a controlled error corrector.

### 6.2 Mathematical Formulation

The final output is a gated combination of coarse and refined predictions:

```
final = coarse + gate × (SAM - coarse)
      = (1 - gate) × coarse + gate × SAM
```

where gate ∈ [gate_min, gate_max] controls the blending.

### 6.3 Uncertainty-Based Gating

The gate value is derived from coarse mask uncertainty:

```
confidence = |2 × coarse - 1|  ∈ [0, 1]
uncertainty = 1 - confidence^γ
gate = gate_min + (gate_max - gate_min) × uncertainty
```

**Interpretation**:
- When coarse ≈ 0.5 (maximum uncertainty): confidence → 0, uncertainty → 1, gate → gate_max
- When coarse ≈ 0 or 1 (high confidence): confidence → 1, uncertainty → 0, gate → gate_min

The parameter γ controls the sensitivity of the gate to confidence levels.

### 6.4 Gate Variants

#### 6.4.1 Uncertainty Gate (Parameter-Free)
Uses only the formula above with no learnable parameters. Simple and interpretable but cannot learn complex correction patterns.

#### 6.4.2 Learned Gate
A small CNN predicts spatially-varying gate values:
```
gate = CNN(coarse_mask, image_features)
```
Architecture: 3 convolutional layers, ~3K parameters. Can learn to identify specific regions requiring correction.

#### 6.4.3 Hybrid Gate
Product of uncertainty and learned gates:
```
gate = uncertainty_gate × learned_gate
```
Combines the safety of uncertainty-based gating with learned flexibility.

### 6.5 Benefits

1. **Conservative Refinement**: Preserves accurate coarse predictions
2. **Targeted Correction**: Focuses refinement on uncertain boundary regions
3. **No Phase 2 Required**: Works with pretrained SAM without finetuning
4. **Interpretable**: Gate values indicate where refinement is applied

---

## 7. Inference-Time Stabilization

### 7.1 Motivation

Even after training, occasional refinement failures may occur—SAM may produce predictions that are worse than the coarse input. Inference-time stabilization provides a safety net without additional training.

### 7.2 Rejection Rules

A set of heuristic rules discard refined predictions when they deviate excessively from the coarse mask:

#### 7.2.1 IoU Consistency
```
if IoU(refined, coarse) < threshold:
    return coarse  # Reject refinement
```
Ensures refinement maintains reasonable agreement with the coarse prediction.

#### 7.2.2 Area Ratio Constraints
```
ratio = area(refined) / area(coarse)
if ratio < min_ratio or ratio > max_ratio:
    return coarse  # Reject refinement
```
Prevents extreme size changes (e.g., refined mask 10× larger than coarse).

#### 7.2.3 Connected Component Count
```
if num_components(refined) > max_components:
    return coarse  # Reject refinement
```
Prevents fragmented predictions with many disconnected regions.

### 7.3 Boundary-Band Fusion

Instead of accepting/rejecting entire predictions, boundary-band fusion restricts refinement to a narrow band around the coarse mask boundary:

```
dilated = morphological_dilation(coarse, iterations=k)
eroded = morphological_erosion(coarse, iterations=k)
boundary_band = dilated - eroded

final = coarse × (1 - boundary_band) + refined × boundary_band
```

**Interpretation**: Interior and exterior regions are taken from the coarse mask (which is already confident there). Only the boundary band—where coarse is uncertain—receives SAM's refinement.

### 7.4 Benefits

1. **Safety Net**: Prevents catastrophic refinement failures
2. **No Training Required**: Applied purely at inference time
3. **Conservative Approach**: Refinement only where most beneficial
4. **Configurable**: Threshold parameters can be tuned per application

---

## 8. Multi-Mask Selection Strategy

### 8.1 SAM's Multi-Mask Output

SAM's mask decoder produces three candidate masks representing different interpretations of the prompt:
- Mask 1: Typically the smallest/most conservative interpretation
- Mask 2: Medium interpretation
- Mask 3: Typically the largest/most inclusive interpretation

Each mask has an associated IoU prediction score indicating estimated quality.

### 8.2 Hard Selection (Non-Differentiable)

Standard SAM inference selects the mask with highest predicted IoU:
```
selected = masks[argmax(iou_predictions)]
```
This argmax operation is non-differentiable.

### 8.3 Soft IoU-Weighted Selection (Differentiable)

UltraRefiner replaces hard selection with soft weighting:
```
weights = softmax(iou_predictions / τ)
refined = Σᵢ weights[i] × masks[i]
```

where τ is a temperature parameter controlling selection sharpness.

**Properties**:
- τ → 0: Approaches hard argmax selection
- τ → ∞: Uniform averaging of all masks
- Intermediate τ: Soft blending weighted by quality

This maintains differentiability while approximating the discrete selection behavior.

---

## 9. Loss Functions

This section provides a comprehensive summary of all loss functions used across the three training phases, with emphasis on Phase 2 and Phase 3 where specialized loss designs are critical to the pipeline's success.

---

### 9.1 Foundational Loss Components

These basic building blocks are combined in various ways across phases.

#### 9.1.1 Binary Cross-Entropy Loss (BCE)

```
L_BCE = -1/N × Σᵢ [yᵢ × log(σ(zᵢ)) + (1 - yᵢ) × log(1 - σ(zᵢ))]
```

where zᵢ is the predicted logit, σ(·) is the sigmoid function, and yᵢ ∈ {0, 1} is the ground truth label for pixel i. This is a pixel-wise classification loss that treats each pixel independently, providing per-pixel gradient signals.

**Note**: When the input is already in probability space [0, 1] (e.g., gated refinement output), we use `binary_cross_entropy` directly instead of `binary_cross_entropy_with_logits`, with clamping to avoid log(0):
```
p_clamped = clamp(p, ε, 1 - ε)
L_BCE_prob = -1/N × Σᵢ [yᵢ × log(p_clamped_i) + (1 - yᵢ) × log(1 - p_clamped_i)]
```

#### 9.1.2 Dice Loss

```
L_Dice = 1 - (2 × Σᵢ pᵢ × yᵢ + ε) / (Σᵢ pᵢ + Σᵢ yᵢ + ε)
```

where pᵢ = σ(zᵢ) is the predicted probability. Dice loss directly optimizes the Dice similarity coefficient (equivalent to the F1 score), making it region-aware. It is less sensitive to class imbalance than BCE because it normalizes by the total predicted and ground truth areas, which is crucial for medical image segmentation where lesions often occupy a small fraction of the image.

**Multi-class variant** (used in Phase 1 for TransUNet with 2-class softmax output):
```
L_Dice_multi = 1/C × Σ_c [1 - (2 × Σᵢ p_ic × y_ic + ε) / (Σᵢ p_ic² + Σᵢ y_ic² + ε)]
```
where C is the number of classes and targets are one-hot encoded.

#### 9.1.3 Focal Loss

```
L_Focal = -α × (1 - p_t)^γ × log(p_t)
```

where p_t = p if y=1, else 1-p. The modulating factor (1 - p_t)^γ down-weights easy examples and focuses training on hard, misclassified pixels.

**Parameters**:
- α = 0.25: Class balancing factor
- γ = 2.0: Focusing parameter (higher γ → more focus on hard examples)

Focal loss is particularly useful for boundary pixels where predictions are uncertain.

#### 9.1.4 IoU Prediction Loss (MSE)

```
L_IoU = MSE(IoU_predicted, IoU_actual) = 1/B × Σ_b (ĉ_b - c_b)²
```

where ĉ_b is SAM's predicted IoU score for sample b, and c_b is the actual IoU between the predicted mask and ground truth. This loss trains SAM's mask quality head to accurately predict which of its candidate masks is best.

---

### 9.2 Phase 1: TransUNet Pre-Training Loss

Phase 1 uses a standard segmentation loss combining cross-entropy with multi-class Dice:

```
L_Phase1 = 0.5 × L_CE(logits, GT) + 0.5 × L_Dice_multi(softmax(logits), GT)
```

where:
- `logits` ∈ ℝ^{B×2×H×W} is TransUNet's raw 2-class output
- `GT` ∈ {0, 1}^{B×H×W} is the ground truth label
- L_CE is standard multi-class cross-entropy loss
- L_Dice_multi is multi-class Dice loss with one-hot encoding

Equal weighting (0.5/0.5) balances pixel-level accuracy (CE) with region-level consistency (Dice).

---

### 9.3 Phase 2: SAM Alignment Loss Functions

Phase 2 uses specialized loss functions that teach SAM not only **what** to predict, but **when to correct** and **when to preserve**. Two variants are used depending on the training approach.

#### 9.3.1 Quality-Aware Loss (Used in Hybrid Training — Primary Approach)

This is the loss function used in the actual hybrid training (50% real predictions + 50% augmented GT):

```
L_Phase2 = L_BCE + L_Dice + λ_change × L_change_penalty
```

**Component 1: BCE Loss (Segmentation Accuracy)**
```
L_BCE = BCE_with_logits(z_refined, GT)
```
where z_refined is SAM's raw logit output and GT is the ground truth mask at 1024×1024.

**Component 2: Dice Loss (Region Consistency)**
```
p_refined = σ(z_refined)
intersection_b = Σ_{h,w} p_refined(b,h,w) × GT(b,h,w)     for each sample b
union_b = Σ_{h,w} p_refined(b,h,w) + Σ_{h,w} GT(b,h,w)
L_Dice = mean_b [1 - (2 × intersection_b + 1) / (union_b + 1)]
```

**Component 3: Change Penalty (Conservative Refinement)**

This is the key innovation in Phase 2's loss. It penalizes SAM for modifying regions where the coarse mask was already correct:

```
coarse_binary = (coarse_mask > 0.5)             — Binarize coarse mask
correct_regions = (coarse_binary == GT)          — Identify correctly predicted pixels
changes = |σ(z_refined) - coarse_binary|         — Magnitude of SAM's modification
L_change_penalty = mean(changes × correct_regions)
```

**Intuition**: If the coarse mask already correctly predicts a pixel (either foreground or background), then SAM should not change it. The penalty is zero for pixels where the coarse mask was wrong (SAM is free to correct those), and proportional to the modification magnitude for pixels where the coarse mask was right.

**Default weight**: λ_change = 0.5

**Total Phase 2 Loss**:
```
L_Phase2 = L_BCE + L_Dice + 0.5 × L_change_penalty
```

This teaches SAM to be a **conservative refiner**: aggressively correcting errors (no penalty) while preserving correct predictions (penalized for changes).

#### 9.3.2 Advanced Quality-Aware Loss (Used in Online Augmented Training — Alternative)

The augmented training variant includes additional components:

```
L_Phase2_advanced = L_BCE + L_Dice + 0.5 × L_Focal + λ_change × L_change_penalty_quality + 0.1 × L_IoU
```

**Component 3 (enhanced): Quality-Weighted Change Penalty**

Instead of binary "correct/incorrect" regions, this variant computes a continuous quality weight based on the input mask's Dice score with GT:

```
input_quality_b = Dice(coarse_mask_b, GT_b)                    — Per-sample quality ∈ [0, 1]
penalty_weight_b = clamp((input_quality_b - 0.6) / 0.4, 0, 1) — Maps [0.6, 1.0] → [0, 1]
change_magnitude_b = mean_{h,w} |σ(z_refined) - coarse_mask|²  — L2 change per sample
L_change_penalty_quality = mean_b (change_magnitude_b × penalty_weight_b)
```

**Behavior by input quality**:
| Input Dice | penalty_weight | Effect |
|------------|---------------|--------|
| ≤ 0.60 (poor) | 0.0 | No penalty — allow SAM to make any changes |
| 0.70 | 0.25 | Mild penalty — some preservation encouraged |
| 0.80 | 0.50 | Moderate penalty |
| 0.90 | 0.75 | Strong penalty — mostly preserve |
| 1.00 (perfect) | 1.0 | Maximum penalty — do not change anything |

This creates a smooth spectrum: poor inputs receive unrestricted correction, while high-quality inputs are strongly preserved.

**Component 4: Focal Loss**

Weighted at 0.5× to focus training on hard boundary pixels where predictions are uncertain:
```
p_t = GT × σ(z) + (1 - GT) × (1 - σ(z))   — Probability of correct class
L_Focal = mean[(1 - p_t)² × BCE_per_pixel]
```

**Component 5: IoU Prediction Loss**

Trains SAM's mask quality head:
```
actual_iou_b = intersection_b / (union_b - intersection_b + 1)
L_IoU = MSE(IoU_predicted, actual_iou)
```

---

### 9.4 Phase 3: End-to-End Joint Optimization Loss

Phase 3's loss supervises both the coarse and refined outputs simultaneously, with asymmetric weighting that emphasizes refinement quality.

#### 9.4.1 EndToEndLoss (Primary Loss Function)

```
L_Phase3 = λ_coarse × L_coarse + λ_refined × L_refined
```

**As used in practice**: λ_coarse = 0.1, λ_refined = 0.9

**Coarse Loss (TransUNet supervision)**:
```
L_coarse = 0.5 × L_CE(logits, GT) + 0.5 × L_Dice_multi(softmax(logits), GT)
```

This is the same loss as Phase 1, maintaining TransUNet's standalone segmentation quality. The 0.1 weight ensures TransUNet doesn't degrade but allows it to adapt outputs for SAM.

**Refined Loss (SAM supervision, standard mode)**:

For standard (non-gated) UltraRefiner, SAM outputs logits:
```
L_refined = L_BCEDice(z_refined, GT_resized)
           = 0.5 × BCE_with_logits(z_refined, GT_resized) + 0.5 × BinaryDiceLoss(σ(z_refined), GT_resized)
```

where GT_resized is the ground truth interpolated to match SAM's output resolution (1024×1024) using nearest-neighbor interpolation.

**Refined Loss (SAM supervision, gated mode)**:

For gated UltraRefiner, the output is already in probability space:
```
L_refined = L_BCEDice_prob(p_refined, GT_resized)
           = 0.5 × BCE(clamp(p_refined, ε, 1-ε), GT_resized) + 0.5 × BinaryDiceLoss(p_refined, GT_resized)
```

#### 9.4.2 Total Phase 3 Loss with All Components

The complete loss including optional protection mechanisms:

```
L_total = λ_coarse × L_coarse(TransUNet, GT)
        + λ_refined × L_refined(SAM, GT_resized)
        + λ_reg × L_weight_regularization(θ_TransUNet)
```

**Weight Regularization** (optional, elastic constraint):
```
L_reg = ||θ_TransUNet - θ_Phase1||²
```
Anchors TransUNet weights to Phase 1 values, preventing catastrophic forgetting during joint training.

#### 9.4.3 Gradient Flow Through the Loss

The total gradient for TransUNet parameters θ is:

```
∂L_total/∂θ = λ_coarse × ∂L_coarse/∂θ
             + λ_refined × ∂L_refined/∂p_refined × ∂p_refined/∂prompts × ∂prompts/∂p_coarse × ∂p_coarse/∂θ
             + λ_reg × 2(θ - θ_Phase1)
```

The first term provides direct supervision on TransUNet's output. The second term (the "refinement gradient") flows backwards through SAM → prompts → coarse mask → TransUNet, teaching TransUNet to produce outputs that are optimal for SAM refinement. The third term acts as a regularizer.

With the actual weights (λ_coarse=0.1, λ_refined=0.9), the refinement gradient dominates, meaning TransUNet learns primarily to produce outputs that maximize SAM's refinement quality rather than standalone segmentation accuracy.

---

### 9.5 Summary: Loss Functions Across All Phases

| Phase | Loss | Formula | Weight | Purpose |
|-------|------|---------|--------|---------|
| **Phase 1** | Cross-Entropy | L_CE(logits, GT) | 0.5 | Pixel-wise classification |
| **Phase 1** | Multi-class Dice | L_Dice(softmax(logits), GT) | 0.5 | Region overlap |
| **Phase 2** | BCE | BCE_with_logits(z, GT) | 1.0 | Pixel-wise refinement accuracy |
| **Phase 2** | Dice | 1 - DiceCoeff(σ(z), GT) | 1.0 | Region overlap for refinement |
| **Phase 2** | Change Penalty | \|σ(z) - coarse\| × I(coarse==GT) | 0.5 | Preserve correct regions |
| **Phase 2** | Focal (optional) | (1-p_t)^γ × BCE | 0.5 | Hard boundary pixels |
| **Phase 2** | IoU Pred (optional) | MSE(IoU_pred, IoU_actual) | 0.1 | Mask quality estimation |
| **Phase 3** | Coarse CE+Dice | 0.5×CE + 0.5×Dice | 0.1 | Maintain TransUNet quality |
| **Phase 3** | Refined BCE+Dice | 0.5×BCE + 0.5×Dice | 0.9 | Optimize refinement output |
| **Phase 3** | Weight Reg (opt.) | \|\|θ - θ_init\|\|² | tunable | Prevent catastrophic forgetting |

**Key Design Principles**:
1. **Phase 2**: BCE + Dice teaches SAM to segment well; Change Penalty teaches SAM to be conservative — only correcting errors, not modifying correct regions
2. **Phase 3**: Asymmetric weighting (0.1/0.9) prioritizes refinement quality, allowing TransUNet to adapt its outputs for SAM's benefit
3. **Across phases**: Dice loss is always included to address the extreme class imbalance in medical segmentation (small lesion vs. large background)

---

## 10. Summary of Innovations

UltraRefiner introduces several key innovations that together enable effective end-to-end refinement:

### 10.1 Differentiable Prompt Extraction
- Soft-argmax point extraction via probability-weighted expectation
- Weighted-statistics box extraction via mean ± k×std
- Direct and Gaussian mask conversion to SAM logit format
- Full gradient flow from SAM output through prompts to coarse network

### 10.2 Three-Phase Curriculum Training
- Phase 1: Independent coarse network training
- Phase 2: SAM distribution alignment with mask augmentation
- Phase 3: End-to-end joint optimization with protection mechanisms

### 10.3 Comprehensive Mask Augmentation
- 12 error types covering realistic failure modes
- Soft mask conversion matching coarse network output characteristics
- Quality-aware loss preventing unnecessary corrections

### 10.4 Differentiable ROI Cropping
- Weighted-statistics ROI extraction
- Grid_sample-based differentiable crop and paste
- 4-16× resolution enhancement for small lesions

### 10.5 Gated Residual Refinement
- Uncertainty-based gating for conservative refinement
- Learned and hybrid gate variants
- Eliminates need for Phase 2 finetuning in some scenarios

### 10.6 Inference-Time Stabilization
- Rejection rules based on IoU, area ratio, component count
- Boundary-band fusion restricting refinement to uncertain regions
- Safety net against refinement failures

### 10.7 TransUNet Protection Mechanisms
- Gradient scaling to slow TransUNet adaptation
- Weight regularization anchoring to Phase 1
- Dual loss supervision maintaining standalone quality
- Two-stage unfreezing for controlled joint optimization

---

## 11. Conclusion

UltraRefiner represents a principled approach to integrating foundation models with task-specific networks in a fully differentiable manner. By solving the prompt extraction non-differentiability problem and addressing the distribution gap through curriculum training, UltraRefiner enables true end-to-end optimization where both networks co-adapt synergistically.

The framework is designed for medical image segmentation—specifically breast ultrasound—but the techniques generalize to any domain requiring refinement of coarse segmentation predictions. The key insight is that foundation models like SAM, despite their impressive zero-shot capabilities, benefit substantially from task-specific adaptation when integrated into an end-to-end learnable pipeline.

Future directions include:
- Extension to 3D medical imaging (CT, MRI volumes)
- Integration with other foundation models beyond SAM
- Application to other medical imaging modalities
- Multi-task learning incorporating detection and classification

---

## References

1. Chen, J., et al. "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation." arXiv:2102.04306 (2021).

2. Kirillov, A., et al. "Segment Anything." arXiv:2304.02643 (2023).

3. Ma, J., et al. "Segment Anything in Medical Images." Nature Communications (2024).

4. Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR (2021).

5. He, K., et al. "Masked Autoencoders Are Scalable Vision Learners." CVPR (2022).

6. Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI (2015).
