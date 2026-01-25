#!/bin/bash
#SBATCH --job-name=sam_finetune_offline
#SBATCH --output=logs/phase2_sam_offline_%j.out
#SBATCH --error=logs/phase2_sam_offline_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/ll5582/3DSAM/envs/ultrarefiner
module load cuda/12.2.0

cd /scratch/ll5582/3DSAM/UltraRefiner
mkdir -p logs

# Configuration
DATA_ROOT="./dataset/augmented_masks"  # Pre-generated augmented data
DATASETS="BUSI BUSBRA BUS BUS_UC BUS_UCLM"
SAM_CHECKPOINT="./pretrained/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE="vit_b"
OUTPUT_DIR="./checkpoints/sam_finetuned_offline"

# Training parameters
EPOCHS=100
BATCH_SIZE=4  # Reduced for memory
LR=1e-4
NUM_WORKERS=8

# Speed optimizations
USE_AMP="--use_amp"
PREFETCH_FACTOR=4

# Phase 3 compatibility
TRANSUNET_IMG_SIZE=224
MASK_PROMPT_STYLE="direct"
USE_ROI_CROP="--use_roi_crop"
ROI_EXPAND_RATIO=0.3

echo "============================================================"
echo "Phase 2: SAM Finetuning with Offline Augmented Data"
echo "============================================================"
echo "  Data root: ${DATA_ROOT}"
echo "  Datasets: ${DATASETS}"
echo "  SAM checkpoint: ${SAM_CHECKPOINT}"
echo "  Output dir: ${OUTPUT_DIR}"
echo ""
echo "Training:"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LR}"
echo "  Mixed Precision: enabled"
echo ""
echo "Phase 3 Compatibility:"
echo "  TransUNet img size: ${TRANSUNET_IMG_SIZE}"
echo "  Mask prompt style: ${MASK_PROMPT_STYLE}"
echo "  ROI cropping: enabled"
echo "============================================================"

python scripts/finetune_sam_offline.py \
    --data_root ${DATA_ROOT} \
    --datasets ${DATASETS} \
    --sam_checkpoint ${SAM_CHECKPOINT} \
    --sam_model_type ${SAM_MODEL_TYPE} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --num_workers ${NUM_WORKERS} \
    --transunet_img_size ${TRANSUNET_IMG_SIZE} \
    --mask_prompt_style ${MASK_PROMPT_STYLE} \
    ${USE_ROI_CROP} \
    --roi_expand_ratio ${ROI_EXPAND_RATIO} \
    ${USE_AMP} \
    --prefetch_factor ${PREFETCH_FACTOR} \
    --output_dir ${OUTPUT_DIR}

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Phase 2 SAM finetuning completed!"
    echo "============================================================"
    echo "Checkpoints saved to: ${OUTPUT_DIR}/"
else
    echo "Failed with exit code $?"
    exit 1
fi

echo "End Time: $(date)"
