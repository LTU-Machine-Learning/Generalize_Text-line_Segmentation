#!/bin/bash
#SBATCH -o /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.out
#SBATCH -e /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.err
#SBATCH -t 0-01:00:00
#SBATCH -C thin
#SBATCH --gpus=1


# mamba init bash
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate pytorch25

# Parameters
file=demo_text_detection_mAP_runtime.py

root=/home/x_gapat/PROJECTS
data_root=/home/x_gapat/PROJECTS/DATASETS


main_script="${root}/codes/Hi-SAM_Doc/${file}"

PATHLOG="${root}/logs/Hi-SAM_Doc/"
output_file="${PATHLOG}/runtime_READ_2016.txt"
output_file2="${PATHLOG}/out.txt"

PRETRAINE_PATH="${PATHLOG}/pretrained_checkpoint"


DATA_IN="${data_root}/READ_2016/test/Images"
DATA_OUT="${root}/logs/Hi-SAM_Doc/sample_output"

export PYTHONPATH=/proj/document_analysis/users/x_gapat/codes/Hi-SAM_Doc/


EXP="60_2025-10-01_ID_14189314"
CHECKPOINT="${PATHLOG}/${EXP}/saved_model"
python "$main_script" \
  --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "$PRETRAINE_PATH" \
  --input "$DATA_IN" \
  --output "$DATA_OUT" \
  --dataset ctw1500 \
  --nms 0.55 0.5 \
  --results_log "$output_file" \
>> "$output_file2"

# python "$main_script" \
#   --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "$PRETRAINE_PATH" \
#   --input "$DATA_IN" \
#   --output "$DATA_OUT" \
#   --dataset ctw1500 \
#   --nms 0.55 0.5 \
#   --results_log "$output_file" \
# >> "$output_file2"

# python "$main_script" \
#   --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "$PRETRAINE_PATH" \
#   --input "$DATA_IN" \
#   --output "$DATA_OUT" \
#   --dataset ctw1500 \
#   --nms 0.55 0.5 \
#   --results_log "$output_file" \
# >> "$output_file2"

# python "$main_script" \
#   --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "$PRETRAINE_PATH" \
#   --input "$DATA_IN" \
#   --output "$DATA_OUT" \
#   --dataset ctw1500 \
#   --nms 0.55 0.5 \
#   --results_log "$output_file" \
# >> "$output_file2"

# python "$main_script" \
#   --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "$PRETRAINE_PATH" \
#   --input "$DATA_IN" \
#   --output "$DATA_OUT" \
#   --dataset ctw1500 \
#   --nms 0.55 0.5 \
#   --results_log "$output_file" \
# >> "$output_file2"