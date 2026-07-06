#!/bin/bash
#SBATCH -o /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.out
#SBATCH -e /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.err
#SBATCH -t 1-00:00:00
#SBATCH -C thin
#SBATCH --gpus=1


# mamba init bash
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate pytorch25

# Parameters
file=demo_text_detection_gridSearch.py

# root=/home/x_gapat/PROJECTS
root="/home/$USER/PROJECTS"
main_script="${root}/codes/Hi-SAM_Doc/${file}"


PATHLOG="${root}/logs/Hi-SAM_Doc/tmp"
echo "path log :"
echo ${PATHLOG}

output_file="${PATHLOG}/out.txt"

export PYTHONPATH=/proj/document_analysis/users/x_gapat/codes/Hi-SAM_Doc/

############################# Train on NorHandV3_331861 #############################
################# H094
PRETRAINE_MODEL=94_2026-04-12_ID_16223622
python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATASETS/NorHandV3_331861/val/Images" \
  --output "${PATHLOG}/H094/NorHandV3_331861/images" \
  --save_boxes_dir "${PATHLOG}/H094/NorHandV3_331861/boxes" \
  --dataset ctw1500 \
  --results_log "${root}/logs/ATS/Hi-SAM/stage1/Hi-SAM_gridsearch.txt" \
  --text TrainOn_NorHandV3_331861_94-TestOn_NorHandV3_331861 \
>> "$output_file"

############################# Train on Nuremberg_Letterbooks/Band3 #############################
# ################# H095
PRETRAINE_MODEL=95_2026-04-17_ID_16360065
python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATASETS/READ_2016/val/Images" \
  --output "${PATHLOG}/H092/READ_2016/images" \
  --save_boxes_dir "${PATHLOG}/H092/READ_2016/boxes" \
  --dataset ctw1500 \
  --results_log "${root}/logs/ATS/Hi-SAM/stage1/Hi-SAM_gridsearch.txt" \
  --text TrainOn_Nuremberg3_95-TestOn_Nuremberg3_95 \
>> "$output_file"

############################# Train on Mix_ReNoNu #############################
################# H096
PRETRAINE_MODEL=96_2026-04-15_ID_16302713
python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATASETS/READ_2016/val/Images" \
  --output "${PATHLOG}/H093/READ_2016/images" \
  --save_boxes_dir "${PATHLOG}/H093/READ_2016/boxes" \
  --dataset ctw1500 \
  --results_log "${root}/logs/ATS/Hi-SAM/stage1/Hi-SAM_gridsearch.txt" \
  --text TrainOn_Mix_ReNoNu_96-TestOn_Mix_ReNoNu_96 \
>> "$output_file"
