#!/bin/bash
#SBATCH -A Berzelius-2025-71
#SBATCH -o /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.out
#SBATCH -e /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.err
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -t 3-00:00:00
#SBATCH --mem=40G
#SBATCH --gpus=4

#Experiment ID
EXPERIMENT_ID="18"

# mamba init bash
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate pytorch25

# Parameters
file=train.py

root=/home/x_gapat/PROJECTS/codes/Hi-SAM_Doc
main_script="${root}/${file}"


echo "Create dir for log"
CURRENTDATE=`date +"%Y-%m-%d"`
echo "currentDate :"
echo $CURRENTDATE
PATHLOG="/home/x_gapat/PROJECTS/logs/Hi-SAM_Doc/${EXPERIMENT_ID}_${CURRENTDATE}_ID_${SLURM_JOB_ID}/"
echo "path log :"
echo ${PATHLOG}
mkdir -p ${PATHLOG}

SAVE_PATH="${PATHLOG}/saved_model/"
mkdir -p ${SAVE_PATH}
output_file="${PATHLOG}/${SLURM_JOB_ID}.txt"

PRETRAINE_PATH="/home/x_gapat/PROJECTS/logs/Hi-SAM_Doc/pretrained_checkpoint"
PRETRAINE_MODEL="${PRETRAINE_PATH}/sam_vit_h_4b8939.pth"

export PYTHONPATH=/proj/document_analysis/users/x_gapat/codes/Hi-SAM_Doc/


# The job
torchrun --nproc_per_node=4 "$main_script" \
  --checkpoint "$PRETRAINE_MODEL" \
  --model-type vit_h \
  --output "$SAVE_PATH" \
  --batch_size_train 1 \
  --lr_drop_epoch 100 \
  --max_epoch_num 800 \
  --train_datasets read2016_train \
  --val_datasets read2016_val \
  --hier_det \
  --skip_words 1 \
  --data_root /proj/document_analysis/users/shared/ \
  --pretrained_path "$PRETRAINE_PATH" \
  --find_unused_params \
  --seed 45 \
  --wandb 1 \
>> "$output_file"