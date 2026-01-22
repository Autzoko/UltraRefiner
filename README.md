# UltraRefiner: End-to-End Differentiable Segmentation Refinement

A unified framework for breast ultrasound segmentation that combines **TransUNet** for initial coarse segmentation with **SAMRefiner** (based on Segment Anything Model) for mask refinement. The entire pipeline is fully differentiable, enabling end-to-end training with gradient flow from SAM back to TransUNet.

## Overview

UltraRefiner implements a three-phase training pipeline:

1. **Phase 1**: Train TransUNet independently on breast ultrasound datasets (per-dataset or combined)
2. **Phase 2**: Finetune SAM using TransUNet predictions as prompts
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
║                              │                                                ║
║                              ▼                                                ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                  INFERENCE: Generate Predictions                        │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║     For each fold's validation set:                                          ║
║                                                                               ║
║  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          ║
║  │  Load Best      │ ─► │  Inference on   │ ─► │  Save Coarse    │          ║
║  │  Checkpoint     │    │  Val Set        │    │  Predictions    │          ║
║  └─────────────────┘    └─────────────────┘    └─────────────────┘          ║
║                                                        │                     ║
║                                                        ▼                     ║
║                              ┌──────────────────────────────────────┐        ║
║                              │  ./predictions/transunet/{dataset}/  │        ║
║                              │    fold_{i}/predictions/*.npy        │        ║
║                              │    fold_{i}/visualizations/*.png     │        ║
║                              └──────────────────────────────────────┘        ║
║                                                                               ║
║                              │                                                ║
║                              ▼                                                ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                     PHASE 2: SAM Finetuning                             │ ║
║  │               (Using Actual TransUNet Predictions)                      │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║     ┌─────────────────┐         ┌─────────────────┐                         ║
║     │  Original Image │         │  TransUNet Pred │                         ║
║     │   (1024×1024)   │         │  (Coarse Mask)  │                         ║
║     └────────┬────────┘         └────────┬────────┘                         ║
║              │                           │                                   ║
║              │     ┌─────────────────────┴─────────────────────┐            ║
║              │     │        Differentiable Prompt Generator    │            ║
║              │     │  ┌─────────┬──────────────┬────────────┐  │            ║
║              │     │  │  Point  │     Box      │    Mask    │  │            ║
║              │     │  │ (soft   │ (threshold   │  (resize   │  │            ║
║              │     │  │ argmax) │  + minmax)   │  to 256²)  │  │            ║
║              │     │  └─────────┴──────────────┴────────────┘  │            ║
║              │     └─────────────────────┬─────────────────────┘            ║
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
║  ┌─────────────────┐                                                        ║
║  │  Refined Mask   │   Loss = BCE-Dice + IoU Prediction Loss                ║
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

### 3. Inference & Visualize TransUNet Predictions

```bash
# Generate predictions for SAM training
python scripts/inference_transunet.py \
    --data_root ./dataset/processed \
    --dataset BUSI \
    --checkpoint_root ./checkpoints/transunet \
    --output_dir ./predictions/transunet \
    --n_splits 5 \
    --visualize
```

### 4. Phase 2: Finetune SAM with Predictions

```bash
# Option A: Use actual TransUNet predictions (recommended)
python scripts/finetune_sam_with_preds.py \
    --data_root ./dataset/processed \
    --pred_root ./predictions/transunet \
    --medsam_checkpoint ./pretrained/medsam_vit_b.pth \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --fold 0

# Option B: Use simulated coarse masks from GT
python scripts/finetune_sam.py \
    --data_root ./dataset/processed \
    --medsam_checkpoint ./pretrained/medsam_vit_b.pth \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --fold 0
```

### 5. Phase 3: End-to-End Training

```bash
python scripts/train_e2e.py \
    --data_root ./dataset/processed \
    --transunet_checkpoint ./checkpoints/transunet/busi/fold_0/best.pth \
    --sam_checkpoint ./checkpoints/sam_finetuned/fold_0/best.pth \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --fold 0 \
    --max_epochs 100
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
│   └── dataset.py                   # Dataset loaders with K-fold CV
├── utils/
│   ├── losses.py                    # Loss functions (Dice, BCE, SAM Loss)
│   └── metrics.py                   # Metrics & TrainingLogger
├── scripts/
│   ├── preprocess_datasets.py       # Data preprocessing
│   ├── train_transunet.py           # Phase 1 training
│   ├── inference_transunet.py       # Generate predictions & visualizations
│   ├── finetune_sam.py              # Phase 2 (simulated masks)
│   ├── finetune_sam_with_preds.py   # Phase 2 (actual predictions)
│   └── train_e2e.py                 # Phase 3 training
├── dataset/
│   ├── raw/                         # Original datasets
│   └── processed/                   # Preprocessed with splits
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
- PyTorch >= 1.10
- CUDA >= 11.0 (recommended)

```bash
pip install -r requirements.txt
```

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
