#!/bin/bash
#SBATCH --job-name=ultrarefiner_phase3_e2e
#SBATCH --output=logs/phase3_e2e_%j.out
#SBATCH --error=logs/phase3_e2e_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=48:00:00

# ============================================================================
# Phase 3: End-to-End Training (Single Dataset, Single Fold)
#
# Usage:
#   sbatch sbatch_phase3_e2e_single.sh BUSI 0
#   sbatch sbatch_phase3_e2e_single.sh BUSBRA 1
# ============================================================================

# Get arguments
DATASET=${1:-BUSI}
FOLD=${2:-0}

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""
echo "Dataset: ${DATASET}"
echo "Fold: ${FOLD}"
echo ""

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/ll5582/3DSAM/envs/segmamba  # Update this path

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment"
    exit 1
fi

echo "Python path: $(which python)"

# Change to working directory
cd /scratch/ll5582/3DSAM/UltraRefiner  # Update this path
echo "Working directory: $(pwd)"
echo ""

# Create logs directory
mkdir -p logs

# ============================================================================
# Configuration
# ============================================================================
DATA_ROOT="./dataset/processed"
TRANSUNET_CHECKPOINT="./checkpoints/transunet/${DATASET}/fold_${FOLD}/best.pth"
SAM_CHECKPOINT="./pretrained/medsam_vit_b.pth"  # Or sam_vit_b_01ec64.pth
OUTPUT_DIR="./checkpoints/ultra_refiner"

# Training parameters
MAX_EPOCHS=100
BATCH_SIZE=8
TRANSUNET_LR=1e-4
SAM_LR=1e-5
N_SPLITS=5

# Check if TransUNet checkpoint exists
if [ ! -f "${TRANSUNET_CHECKPOINT}" ]; then
    echo "Error: TransUNet checkpoint not found: ${TRANSUNET_CHECKPOINT}"
    exit 1
fi

echo "Configuration:"
echo "  Data root: ${DATA_ROOT}"
echo "  Dataset: ${DATASET}"
echo "  Fold: ${FOLD}/${N_SPLITS}"
echo "  TransUNet checkpoint: ${TRANSUNET_CHECKPOINT}"
echo "  SAM checkpoint: ${SAM_CHECKPOINT}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Max epochs: ${MAX_EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo ""

# ============================================================================
# Run E2E Training
# ============================================================================
echo "Starting Phase 3: End-to-End Training..."
echo ""

python scripts/train_e2e.py \
    --data_root ${DATA_ROOT} \
    --datasets ${DATASET} \
    --fold ${FOLD} \
    --n_splits ${N_SPLITS} \
    --transunet_checkpoint ${TRANSUNET_CHECKPOINT} \
    --sam_checkpoint ${SAM_CHECKPOINT} \
    --output_dir ${OUTPUT_DIR}/${DATASET} \
    --max_epochs ${MAX_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --transunet_lr ${TRANSUNET_LR} \
    --sam_lr ${SAM_LR} \
    --num_workers 8

if [ $? -eq 0 ]; then
    echo ""
    echo "Phase 3 E2E training completed successfully!"
    echo "Checkpoint saved to: ${OUTPUT_DIR}/${DATASET}/fold_${FOLD}/"
else
    echo ""
    echo "Phase 3 E2E training failed with exit code $?"
    exit 1
fi

echo ""
echo "End Time: $(date)"
