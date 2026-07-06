#!/bin/bash
#SBATCH -o /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.out
#SBATCH -e /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.err
#SBATCH -t 3-00:00:00
#SBATCH -C thin
#SBATCH --gpus=1


#Experiment ID
EXPERIMENT_ID="60-10_shot"

# mamba init bash
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate pytorch25

# Parameters
file=train3_universal_Few-shot.py

root=/home/x_gapat/PROJECTS
main_script="${root}/codes/Hi-SAM_Doc/${file}"


echo "Create dir for log"
CURRENTDATE=`date +"%Y-%m-%d"`
echo "currentDate :"
echo $CURRENTDATE
PATHLOG="${root}/logs/Hi-SAM_Doc/${EXPERIMENT_ID}_${CURRENTDATE}_ID_${SLURM_JOB_ID}"
echo "path log :"
echo ${PATHLOG}
mkdir -p ${PATHLOG}

SAVE_PATH="${PATHLOG}/saved_model/"
mkdir -p ${SAVE_PATH}
output_file="${PATHLOG}/${SLURM_JOB_ID}.txt"

PRETRAINE_PATH="${root}/logs/Hi-SAM_Doc/pretrained_checkpoint"
# PRETRAINE_MODEL="/home/x_gapat/PROJECTS/logs/Hi-SAM_Doc/60-10_shot_2026-05-01_ID_16472769/saved_model/final_epoch_200.pth"
PRETRAINE_MODEL="/home/x_gapat/PROJECTS/logs/Hi-SAM_Doc/60_2025-10-01_ID_14189314/saved_model/READ_2016_best_mAP.pth"

export PYTHONPATH=/proj/document_analysis/users/x_gapat/codes/Hi-SAM_Doc/

# The job
torchrun --nproc_per_node=1 --master-port=29507 "$main_script" \
  --checkpoint "$PRETRAINE_MODEL" \
  --model-type vit_h \
  --output "$SAVE_PATH" \
  --batch_size_train 1 \
  --lr_drop_epoch 250 \
  --max_epoch_num 300 \
  --lr 1e-4 \
  --train_datasets read2016_train \
  --val_datasets read2016_val \
  --img_ext jpg \
  --hier_det \
  --skip_words 1 \
  --data_root /home/x_gapat/PROJECTS/DATASETS/Mix_ReNoNu \
  --fewshot_dir fewshot10 \
  --pretrained_path "$PRETRAINE_PATH" \
  --distributed True \
  --find_unused_params \
  --seed 45 \
  --wandb 1 \
>> "$output_file"