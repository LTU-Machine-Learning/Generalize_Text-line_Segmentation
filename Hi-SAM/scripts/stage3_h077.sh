#!/bin/bash
#SBATCH -o /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.out
#SBATCH -e /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.err
#SBATCH -t 3-00:00:00
#SBATCH -C thin
#SBATCH --gpus=1


# mamba init bash
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate pytorch25

# Parameters
file=demo_text_detection_mAP_stage3.py

root=/home/x_gapat/PROJECTS
main_script="${root}/codes/Hi-SAM_Doc/${file}"


PATHLOG="${root}/logs/Hi-SAM_Doc/Stage_3"
echo "path log :"
echo ${PATHLOG}

output_file="${PATHLOG}/out.txt"

export PYTHONPATH=/proj/document_analysis/users/x_gapat/codes/Hi-SAM_Doc/

# The job
PRETRAINE_MODEL="${root}/logs/Hi-SAM_Doc/77_2025-12-09_ID_14908887/saved_model/READ_2016_best_mAP.pth"

python "$main_script" \
  --checkpoint "$PRETRAINE_MODEL" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATASETS/NorHandv3_mini_v3/train/Images/" \
  --output "${root}/logs/Hi-SAM_Doc/Stage_3/H077/NorHandv3_mini_v3/train/images" \
  --save_boxes True \
  --save_boxes_dir "${root}/logs/Hi-SAM_Doc/Stage_3/H077/NorHandv3_mini_v3/train/boxes" \
  --dataset ctw1500 \
  --exp_id 77 \
  --nms 0.6 0.6 \
>> "$output_file"

python "$main_script" \
  --checkpoint "$PRETRAINE_MODEL" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATASETS/NorHandv3_mini_v3/val/Images/" \
  --output "${root}/logs/Hi-SAM_Doc/Stage_3/H077/NorHandv3_mini_v3/val/images" \
  --save_boxes True \
  --save_boxes_dir "${root}/logs/Hi-SAM_Doc/Stage_3/H077/NorHandv3_mini_v3/val/boxes" \
  --dataset ctw1500 \
  --exp_id 77 \
  --nms 0.6 0.6 \
>> "$output_file"