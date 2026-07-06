#!/bin/bash
#SBATCH -o /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.out
#SBATCH -e /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.err
#SBATCH -t 0-00:15:00
#SBATCH -C thin
#SBATCH --gpus=1


# mamba init bash
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate pytorch25

# Parameters
file=demo_text_detection_mAP.py

root=/home/x_gapat/PROJECTS
data_root=/proj/document_analysis/users/shared/


main_script="${root}/codes/Hi-SAM_Doc/${file}"

PATHLOG="${root}/logs/Hi-SAM_Doc/"
output_file="${PATHLOG}/results_75.txt"
output_file2="${PATHLOG}/out.txt"

PRETRAINE_PATH="${PATHLOG}/pretrained_checkpoint"


DATA_IN="${data_root}/READ_2016/Test/Images"
DATA_OUT="${root}/logs/Hi-SAM_Doc/sample_output"

export PYTHONPATH=/proj/document_analysis/users/x_gapat/codes/Hi-SAM_Doc/


# Experiment 54 | Baseline
EXP="75_2025-10-24_ID_14441885"
CHECKPOINT="${PATHLOG}/${EXP}/saved_model"
python "$main_script" \
  --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "$PRETRAINE_PATH" \
  --input "$DATA_IN" \
  --output "$DATA_OUT" \
  --dataset ctw1500 \
  --results_log "$output_file" \
>> "$output_file2"

python "$main_script" \
  --checkpoint "${CHECKPOINT}/READ_2016_best.pth" \
  --model-type vit_h \
  --pretrained_path "$PRETRAINE_PATH" \
  --input "$DATA_IN" \
  --output "$DATA_OUT" \
  --dataset ctw1500 \
  --results_log "$output_file" \
>> "$output_file2"

python "$main_script" \
  --checkpoint "${CHECKPOINT}/final_epoch_2000.pth" \
  --model-type vit_h \
  --pretrained_path "$PRETRAINE_PATH" \
  --input "$DATA_IN" \
  --output "$DATA_OUT" \
  --dataset ctw1500 \
  --results_log "$output_file" \
>> "$output_file2"
