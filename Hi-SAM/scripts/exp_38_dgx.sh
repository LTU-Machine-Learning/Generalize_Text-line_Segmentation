#!/bin/bash

#Experiment ID
EXPERIMENT_ID="38"

# Parameters
file=train.py

root=/home/gayapath/PROJECTS
main_script="${root}/codes/Hi-SAM_Doc/${file}"


echo "Create dir for log"
CURRENTDATE=`date +"%Y-%m-%d"`
echo "currentDate :"
echo $CURRENTDATE
PATHLOG="${root}/logs/Hi-SAM_Doc/${EXPERIMENT_ID}_${CURRENTDATE}_ID_${SLURM_JOB_ID}/"
echo "path log :"
echo ${PATHLOG}
mkdir -p ${PATHLOG}

SAVE_PATH="${PATHLOG}/saved_model/"
mkdir -p ${SAVE_PATH}
output_file="${PATHLOG}/${SLURM_JOB_ID}.txt"

PRETRAINE_PATH="${root}/logs/Hi-SAM_Doc/pretrained_checkpoint"
PRETRAINE_MODEL="/home/gayapath/PROJECTS/logs/Hi-SAM_Doc/34_2025-09-07_ID_/saved_model/READ_2016_best_at_784.pth"

export PYTHONPATH=/home/gayapath/PROJECTS/codes/Hi-SAM_Doc/


# The job
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 "$main_script" \
  --checkpoint "$PRETRAINE_MODEL" \
  --model-type vit_h \
  --output "$SAVE_PATH" \
  --batch_size_train 1 \
  --lr_drop_epoch 400 \
  --early_stop 400 \
  --max_epoch_num 200 \
  --lr 1e-5 \
  --train_datasets read2016_train \
  --val_datasets read2016_val \
  --hier_det \
  --skip_words 1 \
  --data_root /home/gayapath/PROJECTS/DATA_DGX1/ \
  --pretrained_path "$PRETRAINE_PATH" \
  --find_unused_params \
  --seed 45 \
  --wandb 1 \
>> "$output_file"