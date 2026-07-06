#!/bin/bash
#SBATCH -o /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.out
#SBATCH -e /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.err
#SBATCH -t 3-00:00:00
#SBATCH -C thin
#SBATCH --gpus=6


#Experiment ID
EXPERIMENT_ID="102"

# mamba init bash
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate pytorch25

# Parameters
file=train3_universal_2.py

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
PRETRAINE_MODEL="${PRETRAINE_PATH}/sam_vit_h_4b8939.pth"

export PYTHONPATH=/proj/document_analysis/users/x_gapat/codes/Hi-SAM_Doc/

# The job
torchrun --nproc_per_node=6 "$main_script" \
  --checkpoint "$PRETRAINE_MODEL" \
  --model-type vit_h \
  --output "$SAVE_PATH" \
  --batch_size_train 1 \
  --lr_drop_patience 200 \
  --early_stop_patience 200 \
  --max_epoch_num 3000 \
  --lr 1e-4 \
  --train_datasets read2016_train \
  --val_datasets read2016_val \
  --img_ext jpg \
  --hier_det \
  --skip_words 1 \
  --data_root /home/x_gapat/PROJECTS/DATASETS/Mix_ReNoNu \
  --pretrained_path "$PRETRAINE_PATH" \
  --distributed True \
  --find_unused_params \
  --seed 75 \
  --wandb 1 \
>> "$output_file"