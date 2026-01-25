#!/bin/bash
#SBATCH --job-name=gen_transunet_preds
#SBATCH --output=logs/gen_transunet_preds_%j.out
#SBATCH --error=logs/gen_transunet_preds_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/ll5582/3DSAM/envs/ultrarefiner  # Update this path
module load cuda/12.2.0

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment"
    exit 1
fi

echo "Python path: $(which python)"
echo ""

# Change to working directory
cd /scratch/ll5582/3DSAM/UltraRefiner  # Update this path
echo "Working directory: $(pwd)"
echo ""

mkdir -p logs

# ============================================================================
# Generate TransUNet Predictions for Hybrid Training
# ============================================================================
echo "Generating TransUNet predictions on training data..."
echo ""

# Configuration
DATA_ROOT="./dataset/processed"
# Note: Checkpoint dirs use lowercase (busi, busbra, bus, bus_uc, bus_uclm)
# but dataset dirs use uppercase (BUSI, BUSBRA, BUS, BUS_UC, BUS_UCLM)
DATASETS="BUSI BUSBRA BUS BUS_UC BUS_UCLM"
CHECKPOINT_DIR="./checkpoints/transunet"  # Directory with trained TransUNet models
OUTPUT_DIR="./dataset/transunet_preds"
IMG_SIZE=224

# Use ensemble of all folds for better predictions
USE_ALL_FOLDS="--use_all_folds"
# Or use a specific fold:
# FOLD=0

echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "  Data root: ${DATA_ROOT}"
echo "  Datasets: ${DATASETS}"
echo "  Checkpoint dir: ${CHECKPOINT_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Image size: ${IMG_SIZE}"
echo "  Using: All folds (ensemble)"
echo "============================================================"
echo ""

# Build command
CMD="python scripts/generate_transunet_predictions.py \
    --data_root ${DATA_ROOT} \
    --datasets ${DATASETS} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --img_size ${IMG_SIZE} \
    ${USE_ALL_FOLDS}"

echo "Running command:"
echo "${CMD}"
echo ""

eval ${CMD}

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "TransUNet predictions generated successfully!"
    echo "============================================================"
    echo "Output saved to: ${OUTPUT_DIR}/"
    echo ""
    echo "Directory structure:"
    echo "  ${OUTPUT_DIR}/"
    echo "  └── {dataset}/"
    echo "      └── train/"
    echo "          ├── images/       (symlinks to original)"
    echo "          ├── masks/        (symlinks to GT)"
    echo "          └── coarse_masks/ (TransUNet soft predictions as .npy)"
    echo ""
    echo "Next step: Run hybrid SAM finetuning with:"
    echo "  sbatch sbatch/sbatch_phase2_finetune_sam_hybrid.sh"
else
    echo ""
    echo "Failed to generate predictions with exit code $?"
    exit 1
fi

echo ""
echo "End Time: $(date)"
