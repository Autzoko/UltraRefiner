#!/bin/bash
#SBATCH --job-name=ultrarefiner_phase2_sam
#SBATCH --output=logs/phase2_sam_finetune_%j.out
#SBATCH --error=logs/phase2_sam_finetune_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=95:00:00

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
conda activate /scratch/ll5582/3DSAM/envs/segmamba  # Update this path to your environment

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
# Phase 2: Finetune SAM with Augmented Data
# ============================================================================
echo "Starting Phase 2: SAM Finetuning with Augmented Data..."
echo ""

# Configuration - modify these as needed
DATA_ROOT="./dataset/augmented"
DATASET="BUS_BUSBRA_BUSI_BUS_UC_BUS_UCLM"  # Combined dataset folder name
SAM_CHECKPOINT="./pretrained/medsam_vit_b.pth"
SAM_MODEL_TYPE="vit_b"
OUTPUT_DIR="./checkpoints/sam_finetuned"
EPOCHS=50
BATCH_SIZE=4
LR=1e-4
CHANGE_PENALTY_WEIGHT=0.1
NUM_WORKERS=8

# Optional: Resume from checkpoint (uncomment to use)
# RESUME_CHECKPOINT="./checkpoints/sam_finetuned/${DATASET}/best.pth"

echo "Configuration:"
echo "  Data root: ${DATA_ROOT}"
echo "  Dataset: ${DATASET}"
echo "  SAM checkpoint: ${SAM_CHECKPOINT}"
echo "  SAM model type: ${SAM_MODEL_TYPE}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LR}"
echo "  Change penalty weight: ${CHANGE_PENALTY_WEIGHT}"
echo ""

# Build command
CMD="python scripts/finetune_sam_augmented.py \
    --data_root ${DATA_ROOT} \
    --dataset ${DATASET} \
    --sam_checkpoint ${SAM_CHECKPOINT} \
    --sam_model_type ${SAM_MODEL_TYPE} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --change_penalty_weight ${CHANGE_PENALTY_WEIGHT} \
    --num_workers ${NUM_WORKERS} \
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
    echo "Phase 2 SAM finetuning completed successfully!"
    echo "Checkpoints saved to: ${OUTPUT_DIR}/${DATASET}/"
    echo "  - best.pth: Full refiner state (for resuming)"
    echo "  - best_sam.pth: SAM weights (for Phase 3)"
else
    echo ""
    echo "Phase 2 SAM finetuning failed with exit code $?"
    exit 1
fi

echo ""
echo "End Time: $(date)"
