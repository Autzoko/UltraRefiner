#!/bin/bash
# ============================================================================
# Phase 3: Launch E2E Training for All Datasets and Folds
#
# This script submits 25 jobs (5 datasets × 5 folds) to SLURM
#
# Usage:
#   bash sbatch/sbatch_phase3_e2e_all.sh           # Submit all jobs
#   bash sbatch/sbatch_phase3_e2e_all.sh --dry-run # Show commands without submitting
# ============================================================================

DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
    echo ""
fi

# Datasets and folds
DATASETS=("BUSI" "BUSBRA" "BUS" "BUS_UC" "BUS_UCLM")
FOLDS=(0 1 2 3 4)

# Working directory
WORK_DIR="/scratch/ll5582/3DSAM/UltraRefiner"  # Update this path
SBATCH_SCRIPT="${WORK_DIR}/sbatch/sbatch_phase3_e2e_single.sh"

# Check if sbatch script exists
if [ ! -f "${SBATCH_SCRIPT}" ]; then
    echo "Error: sbatch script not found: ${SBATCH_SCRIPT}"
    exit 1
fi

echo "Submitting Phase 3 E2E training jobs..."
echo "Datasets: ${DATASETS[@]}"
echo "Folds: ${FOLDS[@]}"
echo "Total jobs: $((${#DATASETS[@]} * ${#FOLDS[@]}))"
echo ""

JOB_COUNT=0

for DATASET in "${DATASETS[@]}"; do
    for FOLD in "${FOLDS[@]}"; do
        JOB_NAME="e2e_${DATASET}_f${FOLD}"

        # Check if TransUNet checkpoint exists
        CHECKPOINT="${WORK_DIR}/checkpoints/transunet/${DATASET}/fold_${FOLD}/best.pth"

        if [ ! -f "${CHECKPOINT}" ]; then
            echo "[SKIP] ${DATASET} fold ${FOLD}: TransUNet checkpoint not found"
            continue
        fi

        if [ "$DRY_RUN" = true ]; then
            echo "[DRY-RUN] sbatch --job-name=${JOB_NAME} ${SBATCH_SCRIPT} ${DATASET} ${FOLD}"
        else
            echo "[SUBMIT] ${DATASET} fold ${FOLD}"
            sbatch --job-name=${JOB_NAME} ${SBATCH_SCRIPT} ${DATASET} ${FOLD}
        fi

        JOB_COUNT=$((JOB_COUNT + 1))
    done
done

echo ""
echo "Submitted ${JOB_COUNT} jobs"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: ${WORK_DIR}/logs/"
