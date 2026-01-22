# UltraRefiner: End-to-End Differentiable Segmentation Refinement

A unified framework for breast ultrasound segmentation that combines **TransUNet** for initial coarse segmentation with **SAMRefiner** (based on Segment Anything Model) for mask refinement. The entire pipeline is fully differentiable, enabling end-to-end training with gradient flow from SAM back to TransUNet.

## Overview

UltraRefiner implements a three-phase training pipeline:

1. **Phase 1**: Train TransUNet independently on breast ultrasound datasets
2. **Phase 2**: Finetune SAM (from MedSAM checkpoint) for mask refinement
3. **Phase 3**: End-to-end training with gradients flowing from SAMRefiner to TransUNet

### Architecture

```
Input Image
    │
    ▼
┌─────────────┐
│  TransUNet  │  ← Phase 1 training
│  (R50-ViT)  │
└─────────────┘
    │
    ▼ (Soft Probability Mask)
┌─────────────────────────────┐
│   Differentiable Prompts    │
│  ┌─────────┬─────┬───────┐  │
│  │  Point  │ Box │ Mask  │  │
│  └─────────┴─────┴───────┘  │
└─────────────────────────────┘
    │
    ▼
┌─────────────┐
│ SAMRefiner  │  ← Phase 2 training
│ (MedSAM)    │
└─────────────┘
    │
    ▼
Refined Mask
```

### Key Features

- **Fully Differentiable Pipeline**: Enables end-to-end optimization
- **Differentiable Prompt Generation**: Soft-argmax based point extraction, differentiable box estimation
- **Multi-Dataset Support**: BUSI, BUSBRA, BUS, BUS_UC, BUS_UCLM
- **MedSAM Integration**: Leverages medical image-specific SAM pretraining
- **Modular Design**: Each component can be trained independently or jointly

## Project Structure

```
UltraRefiner/
├── configs/
│   ├── __init__.py
│   └── config.py              # Configuration management
├── models/
│   ├── __init__.py
│   ├── transunet/             # TransUNet model
│   │   ├── vit_seg_modeling.py
│   │   ├── vit_seg_configs.py
│   │   └── vit_seg_modeling_resnet_skip.py
│   ├── sam/                   # SAM model
│   │   ├── sam.py
│   │   ├── image_encoder.py
│   │   ├── mask_decoder.py
│   │   ├── prompt_encoder.py
│   │   └── build_sam.py
│   ├── sam_refiner.py         # Differentiable SAM Refiner
│   └── ultra_refiner.py       # End-to-end model
├── data/
│   ├── __init__.py
│   └── dataset.py             # Unified dataset loader
├── utils/
│   ├── __init__.py
│   ├── losses.py              # Loss functions
│   └── metrics.py             # Evaluation metrics
├── scripts/
│   ├── train_transunet.py     # Phase 1 training
│   ├── finetune_sam.py        # Phase 2 training
│   ├── train_e2e.py           # Phase 3 training
│   └── inference.py           # Inference script
├── checkpoints/               # Model checkpoints
└── requirements.txt
```

## Installation

```bash
# Clone the repository
cd UltraRefiner

# Install dependencies
pip install -r requirements.txt

# Install FastGeodis for distance transform
pip install FastGeodis
```

## Data Preparation

Organize your datasets as follows:

```
data/
├── BUSI/
│   ├── benign/
│   │   ├── benign (1).png
│   │   ├── benign (1)_mask.png
│   │   └── ...
│   └── malignant/
│       └── ...
├── BUSBRA/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── val/
│       └── ...
└── ... (other datasets)
```

## Training

### Phase 1: Train TransUNet

```bash
python scripts/train_transunet.py \
    --data_root ./data \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --vit_name R50-ViT-B_16 \
    --vit_pretrained ./pretrained/R50+ViT-B_16.npz \
    --img_size 224 \
    --batch_size 24 \
    --max_epochs 150 \
    --output_dir ./checkpoints/transunet \
    --exp_name transunet_bus
```

### Phase 2: Finetune SAM

```bash
python scripts/finetune_sam.py \
    --data_root ./data \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --sam_model_type vit_b \
    --medsam_checkpoint ./pretrained/medsam_vit_b.pth \
    --batch_size 4 \
    --max_epochs 100 \
    --output_dir ./checkpoints/sam_finetuned \
    --exp_name sam_finetune_bus
```

### Phase 3: End-to-End Training

```bash
python scripts/train_e2e.py \
    --data_root ./data \
    --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
    --transunet_checkpoint ./checkpoints/transunet/best.pth \
    --sam_checkpoint ./checkpoints/sam_finetuned/best.pth \
    --batch_size 8 \
    --max_epochs 100 \
    --transunet_lr 1e-4 \
    --sam_lr 1e-5 \
    --coarse_loss_weight 0.3 \
    --refined_loss_weight 0.7 \
    --output_dir ./checkpoints/ultra_refiner \
    --exp_name ultra_refiner_e2e
```

## Inference

```bash
# Full pipeline inference
python scripts/inference.py \
    --mode full \
    --model_checkpoint ./checkpoints/ultra_refiner/best.pth \
    --image_path ./test_image.png \
    --output_dir ./results

# TransUNet only inference
python scripts/inference.py \
    --mode transunet \
    --transunet_checkpoint ./checkpoints/transunet/best.pth \
    --image_dir ./test_images \
    --output_dir ./results
```

## Model Components

### TransUNet
- **Architecture**: ResNet50 + ViT-B/16 hybrid encoder with CNN decoder
- **Input**: 224×224 grayscale images
- **Output**: Probability maps for each class

### SAMRefiner
- **Architecture**: SAM's prompt encoder + mask decoder
- **Input**: 1024×1024 images + differentiable prompts
- **Prompts**:
  - Point prompts: Soft-argmax extracted positive/negative points
  - Box prompts: Differentiable bounding box from mask
  - Mask prompts: Scaled and transformed coarse mask

### Differentiable Prompt Generation
The key innovation is making prompt generation differentiable:

1. **Point Prompts**: Uses soft-argmax to extract the centroid as a weighted average
2. **Box Prompts**: Extracts bounding box coordinates from thresholded mask
3. **Mask Prompts**: Directly passes soft probability maps as mask input

## Loss Functions

- **Phase 1**: CrossEntropy + Dice Loss for multi-class segmentation
- **Phase 2**: BCE-Dice Loss + IoU Prediction Loss for SAM training
- **Phase 3**: Combined weighted loss:
  ```
  L_total = λ_coarse * L_TransUNet + λ_refined * L_SAM
  ```

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.10
- CUDA ≥ 11.0 (recommended)
- See `requirements.txt` for complete dependencies

## Pretrained Weights

Download the following pretrained weights:

1. **ViT weights for TransUNet**: Download from [Google Cloud](https://console.cloud.google.com/storage/browser/vit_models/imagenet21k)
2. **MedSAM checkpoint**: Download from [MedSAM GitHub](https://github.com/bowang-lab/MedSAM)

## Citation

If you use this code, please cite:

```bibtex
@article{ultrarefiner2024,
  title={UltraRefiner: End-to-End Differentiable Segmentation Refinement for Breast Ultrasound},
  author={},
  journal={},
  year={2024}
}
```

## References

- TransUNet: [Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306)
- SAMRefiner: [SAMRefiner: A General-Purpose Mask Refiner](https://github.com/linyq2117/SAMRefiner)
- MedSAM: [Segment Anything in Medical Images](https://github.com/bowang-lab/MedSAM)

## License

This project is for research purposes only.
