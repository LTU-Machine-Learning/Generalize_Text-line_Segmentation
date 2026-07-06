#!/bin/bash

#Experiment ID
EXPERIMENT_ID="42"

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
PRETRAINE_MODEL="${PRETRAINE_PATH}/sam_vit_h_4b8939.pth"

export PYTHONPATH=/home/gayapath/PROJECTS/codes/Hi-SAM_Doc/

# The job
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 "$main_script" \
  --checkpoint "$PRETRAINE_MODEL" \
  --model-type vit_h \
  --output "$SAVE_PATH" \
  --batch_size_train 1 \
  --lr_drop_epoch 500 \
  --early_stop 200 \
  --max_epoch_num 800 \
  --train_datasets read2016_train \
  --val_datasets read2016_val \
  --hier_det \
  --skip_words 1 \
  --data_root /home/gayapath/PROJECTS/DATA_DGX2/ \
  --pretrained_path "$PRETRAINE_PATH" \
  --distributed True \
  --find_unused_params \
  --seed 45 \
  --wandb 1 \
>> "$output_file"