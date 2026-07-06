#!/bin/bash
#SBATCH -A Berzelius-2025-71
#SBATCH -o /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.out
#SBATCH -e /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.err
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -t 1-00:00:00
#SBATCH --mem=40G
#SBATCH --gpus=1

# mamba init bash
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate pytorch25

# Parameters
EXPERIMENT_ID="26"
LOG_FOLDER="18_2025-08-26_ID_13983323"

file=demo_text_detection_mAP.py

root=/home/x_gapat/PROJECTS/codes/Hi-SAM_Doc
main_script="${root}/${file}"

PATHLOG="/home/x_gapat/PROJECTS/logs/Hi-SAM_Doc/"
output_file="${PATHLOG}/results.txt"
output_file2="${PATHLOG}/out.txt"

PRETRAINE_PATH="/home/x_gapat/PROJECTS/logs/Hi-SAM_Doc/pretrained_checkpoint"
PRETRAINE_MODEL="/home/x_gapat/PROJECTS/logs/Hi-SAM_Doc/${LOG_FOLDER}/saved_model/READ_2016_best.pth"

DATA_IN="/home/x_gapat/PROJECTS/DATASETS/READ_2016/Test/Images"
DATA_OUT="/home/x_gapat/PROJECTS/logs/Hi-SAM_Doc/sample_output"

export PYTHONPATH=/proj/document_analysis/users/x_gapat/codes/Hi-SAM_Doc/


# The job
srun  python "$main_script" \
  --checkpoint "$PRETRAINE_MODEL" \
  --model-type vit_h \
  --pretrained_path "$PRETRAINE_PATH" \
  --input "$DATA_IN" \
  --output "$DATA_OUT" \
  --dataset ctw1500 \
  --results_log "$output_file" \
  --exp_id "$EXPERIMENT_ID" \
>> "$output_file2"