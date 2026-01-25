#!/bin/bash
#SBATCH --job-name=gen_aug_masks
#SBATCH --output=logs/gen_aug_masks_%j.out
#SBATCH --error=logs/gen_aug_masks_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00

echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/ll5582/3DSAM/envs/ultrarefiner

cd /scratch/ll5582/3DSAM/UltraRefiner
mkdir -p logs

# Configuration
DATA_ROOT="./dataset/processed"
DATASETS="BUSI BUSBRA BUS BUS_UC BUS_UCLM"
OUTPUT_DIR="./dataset/augmented_masks"
NUM_AUGMENTATIONS=5  # 5 augmented versions per sample = 5x data
AUGMENTOR_PRESET="default"
NUM_WORKERS=16

echo "============================================================"
echo "Generating Offline Augmented Masks"
echo "============================================================"
echo "  Data root: ${DATA_ROOT}"
echo "  Datasets: ${DATASETS}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Augmentations per sample: ${NUM_AUGMENTATIONS}"
echo "  Preset: ${AUGMENTOR_PRESET}"
echo "============================================================"

python scripts/generate_augmented_masks.py \
    --data_root ${DATA_ROOT} \
    --datasets ${DATASETS} \
    --output_dir ${OUTPUT_DIR} \
    --num_augmentations ${NUM_AUGMENTATIONS} \
    --augmentor_preset ${AUGMENTOR_PRESET} \
    --use_fast_soft_mask \
    --num_workers ${NUM_WORKERS}

if [ $? -eq 0 ]; then
    echo ""
    echo "Augmented masks generated successfully!"
    echo "Output: ${OUTPUT_DIR}"
    echo ""
    echo "Next step: Train SAM with:"
    echo "  sbatch sbatch/sbatch_phase2_finetune_sam_offline.sh"
else
    echo "Failed with exit code $?"
    exit 1
fi

echo "End Time: $(date)"
