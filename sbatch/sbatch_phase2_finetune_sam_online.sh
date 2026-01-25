#!/bin/bash
#SBATCH --job-name=ultrarefiner_phase2_online
#SBATCH --output=logs/phase2_sam_online_%j.out
#SBATCH --error=logs/phase2_sam_online_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=90:00:00

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Activate conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/ll5582/3DSAM/envs/ultrarefiner  # Update this path to your environment
module load cuda/12.2.0

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment"
    exit 1
fi

echo "Conda environment activated successfully"
echo "Python path: $(which python)"
echo ""

# Change to working directory
cd /scratch/ll5582/3DSAM/UltraRefiner  # Update this path to your UltraRefiner directory
echo "Changed to directory: $(pwd)"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# ============================================================================
# Phase 2: Finetune SAM with Online Mask Augmentation
# ============================================================================
echo "Starting Phase 2: SAM Finetuning with Online Augmentation..."
echo ""

# Configuration - modify these as needed
DATA_ROOT="./dataset/processed"
DATASETS="BUSI BUSBRA BUS BUS_UC BUS_UCLM"  # Space-separated list of datasets
SAM_CHECKPOINT="./pretrained/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE="vit_b"
OUTPUT_DIR="./checkpoints/sam_finetuned_online"

# Training parameters
EPOCHS=300
BATCH_SIZE=4
LR=1e-4
NUM_WORKERS=8

# Augmentation parameters
AUGMENTOR_PRESET="default"  # Options: default, mild, severe, boundary_focus, structural
SOFT_MASK_PROB=0.8
CHANGE_PENALTY_WEIGHT=0.5

# Phase 3 compatibility settings
TRANSUNET_IMG_SIZE=224
MASK_PROMPT_STYLE="direct"
USE_ROI_CROP="--use_roi_crop"
ROI_EXPAND_RATIO=0.5

# Optional: Resume from checkpoint (uncomment to use)
# RESUME_CHECKPOINT="./checkpoints/sam_finetuned_online/BUSI_BUSBRA_BUS_BUS_UC_BUS_UCLM/best.pth"

echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "  Data root: ${DATA_ROOT}"
echo "  Datasets: ${DATASETS}"
echo "  SAM checkpoint: ${SAM_CHECKPOINT}"
echo "  SAM model type: ${SAM_MODEL_TYPE}"
echo "  Output dir: ${OUTPUT_DIR}"
echo ""
echo "Training:"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LR}"
echo ""
echo "Augmentation:"
echo "  Preset: ${AUGMENTOR_PRESET}"
echo "  Soft mask probability: ${SOFT_MASK_PROB}"
echo "  Change penalty weight: ${CHANGE_PENALTY_WEIGHT}"
echo ""
echo "Phase 3 Compatibility:"
echo "  TransUNet img size: ${TRANSUNET_IMG_SIZE}"
echo "  Mask prompt style: ${MASK_PROMPT_STYLE}"
echo "  ROI cropping: enabled"
echo "  ROI expand ratio: ${ROI_EXPAND_RATIO}"
echo "============================================================"
echo ""

# Build command
CMD="python scripts/finetune_sam_online.py \
    --data_root ${DATA_ROOT} \
    --datasets ${DATASETS} \
    --sam_checkpoint ${SAM_CHECKPOINT} \
    --sam_model_type ${SAM_MODEL_TYPE} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --num_workers ${NUM_WORKERS} \
    --augmentor_preset ${AUGMENTOR_PRESET} \
    --soft_mask_prob ${SOFT_MASK_PROB} \
    --change_penalty_weight ${CHANGE_PENALTY_WEIGHT} \
    --transunet_img_size ${TRANSUNET_IMG_SIZE} \
    --mask_prompt_style ${MASK_PROMPT_STYLE} \
    ${USE_ROI_CROP} \
    --roi_expand_ratio ${ROI_EXPAND_RATIO} \
    --output_dir ${OUTPUT_DIR}"

# Add resume flag if checkpoint exists
if [ -n "${RESUME_CHECKPOINT}" ] && [ -f "${RESUME_CHECKPOINT}" ]; then
    echo "Resuming from checkpoint: ${RESUME_CHECKPOINT}"
    CMD="${CMD} --resume ${RESUME_CHECKPOINT}"
fi

# Run training
echo "Running command:"
echo "${CMD}"
echo ""

eval ${CMD}

# Check training exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Phase 2 SAM finetuning (online augmentation) completed!"
    echo "============================================================"
    echo "Checkpoints saved to: ${OUTPUT_DIR}/"
    echo "  - best.pth: Full refiner state (for resuming)"
    echo "  - best_sam.pth: SAM weights (for Phase 3)"
    echo ""
    echo "12 Error Types Applied:"
    echo "  1. Identity/Near-Perfect (15%)"
    echo "  2. Over-Segmentation (17%)"
    echo "  3. Giant Over-Segmentation (10%)"
    echo "  4. Under-Segmentation (17%)"
    echo "  5. Missing Chunk (12%)"
    echo "  6. Internal Holes (10%)"
    echo "  7. Bridge/Adhesion (10%)"
    echo "  8. False Positive Islands (17%)"
    echo "  9. Fragmentation (10%)"
    echo "  10. Shift/Wrong Location (10%)"
    echo "  11. Empty Prediction (4%)"
    echo "  12. Noise-Only Scatter (3%)"
else
    echo ""
    echo "Phase 2 SAM finetuning failed with exit code $?"
    exit 1
fi

echo ""
echo "End Time: $(date)"
