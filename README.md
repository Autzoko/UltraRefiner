# UltraRefiner: End-to-End Differentiable Segmentation Refinement

A unified framework for breast ultrasound segmentation that combines **TransUNet** for initial coarse segmentation with **SAMRefiner** (based on Segment Anything Model) for mask refinement. The entire pipeline is fully differentiable, enabling end-to-end training with gradient flow from SAM back to TransUNet.

## Overview

UltraRefiner implements a three-phase training pipeline:

1. **Phase 1**: Train TransUNet independently on breast ultrasound datasets (per-dataset or combined)
2. **Phase 2**: Finetune SAM using augmented GT masks that simulate realistic segmentation failures
3. **Phase 3**: End-to-end training with gradients flowing from SAMRefiner to TransUNet

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
║  ┌─────────────────┐                                                        ║
║  │   Checkpoints   │   ./checkpoints/sam_finetuned/fold_{i}/best.pth        ║
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
Training mixes perfect masks (10%), mildly corrupted masks (55%), and heavily corrupted masks (35%). A change-penalty term weighted by input mask quality encourages the Refiner to preserve already-good predictions while strongly correcting poor ones.

```bash
# Step 1: Generate augmented training data from all datasets combined
python scripts/generate_augmented_data.py \
    --data_root ./dataset/processed \
    --output_dir ./dataset/augmented \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --target_samples 100000 \
    --use_sdf \
    --num_workers 8

# Step 2: Finetune SAM with augmented data
# Note: --dataset should match the combined folder name (sorted alphabetically, joined by _)
python scripts/finetune_sam_augmented.py \
    --data_root ./dataset/augmented \
    --dataset BUS_BUSBRA_BUSI_BUS_UC_BUS_UCLM \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --sam_model_type vit_b \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --change_penalty_weight 0.1 \
    --output_dir ./checkpoints/sam_finetuned

# Optional: Use curriculum learning (easy to hard)
python scripts/finetune_sam_augmented.py \
    --data_root ./dataset/augmented \
    --dataset BUS_BUSBRA_BUSI_BUS_UC_BUS_UCLM \
    --sam_checkpoint ./pretrained/medsam_vit_b.pth \
    --epochs 50 \
    --batch_size 4 \
    --change_penalty_weight 0.1 \
    --curriculum \
    --output_dir ./checkpoints/sam_finetuned
```

**Option B: Use Actual TransUNet Predictions (Alternative)**

Alternatively, finetune SAM using actual TransUNet predictions from Phase 1. This requires generating predictions first:

```bash
# Step 1: Generate TransUNet predictions
python scripts/inference_transunet.py \
    --data_root ./dataset/processed \
    --dataset BUSI \
    --checkpoint_root ./checkpoints/transunet \
    --output_dir ./predictions/transunet \
    --n_splits 5

# Step 2: Finetune SAM with predictions
python scripts/finetune_sam_with_preds.py \
    --data_root ./dataset/processed \
    --pred_root ./predictions/transunet \
    --medsam_checkpoint ./pretrained/medsam_vit_b.pth \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --fold 0
```

### 4. Phase 3: End-to-End Training

```bash
python scripts/train_e2e.py \
    --data_root ./dataset/processed \
    --transunet_checkpoint ./checkpoints/transunet/combined/fold_0/best.pth \
    --sam_checkpoint ./checkpoints/sam_finetuned/best.pth \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --fold 0 \
    --max_epochs 100 \
    --batch_size 8 \
    --transunet_lr 1e-4 \
    --sam_lr 1e-5
```

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
