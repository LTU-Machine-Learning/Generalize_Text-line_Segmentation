#!/bin/bash
#SBATCH -o /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.out
#SBATCH -e /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.err
#SBATCH -t 3-00:00:00
#SBATCH -C thin
#SBATCH --gpus=6


#Experiment ID
EXPERIMENT_ID="92"

# mamba init bash
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate pytorch25

# Parameters
file=train3_universal_htrScore.py

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
PRETRAINE_MODEL="${PRETRAINE_PATH}/hi_sam_h.pth"

export PYTHONPATH=/proj/document_analysis/users/x_gapat/codes/Hi-SAM_Doc/

# The job
torchrun --nproc_per_node=6 "$main_script" \
  --checkpoint "$PRETRAINE_MODEL" \
  --model-type vit_h \
  --output "$SAVE_PATH" \
  --batch_size_train 1 \
  --lr_drop_patience 200 \
  --early_stop_patience 200 \
  --max_epoch_num 1500 \
  --train_datasets read2016_train \
  --val_datasets read2016_val \
  --img_ext JPG \
  --hier_det \
  --skip_words 1 \
  --data_root "${root}/DATASETS/READ_2016" \
  --pretrained_path "$PRETRAINE_PATH" \
  --distributed True \
  --find_unused_params \
  --htr_charset charset_read2016.txt \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/14_2025-12-09_ID_14908711/crnn_best.torch" \
  --htr_height_max 64 \
  --htr_width_max 1024 \
  --htr_bb_sorting multi-col-sorting \
  --seed 15 \
  --wandb 1 \
>> "$output_file"