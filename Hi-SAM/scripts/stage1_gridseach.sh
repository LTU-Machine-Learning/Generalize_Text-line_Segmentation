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
file=demo_text_detection_gridSearch.py

# root=/home/x_gapat/PROJECTS
root="/home/$USER/PROJECTS"
main_script="${root}/codes/Hi-SAM_Doc/${file}"


PATHLOG="${root}/logs/Hi-SAM_Doc/tmp"
echo "path log :"
echo ${PATHLOG}

output_file="${PATHLOG}/out.txt"

export PYTHONPATH=/proj/document_analysis/users/x_gapat/codes/Hi-SAM_Doc/

# ############################# Train on READ #############################

PRETRAINE_MODEL=103_2026-05-13_ID_16519201
python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATASETS/Mix_ReNoNu/val/Images" \
  --output "${PATHLOG}/H103/Mix_ReNoNu/images" \
  --save_boxes_dir "${PATHLOG}/H103/Mix_ReNoNu/boxes" \
  --dataset ctw1500 \
  --results_log "${root}/logs/ATS/Hi-SAM/stage1/Hi-SAM_gridsearch.txt" \
  --text TrainOn_MIX_RENONU_103-TestOn_MIX_RENONU \
>> "$output_file"

# PRETRAINE_MODEL=91_2026-02-05_ID_15442927
# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_CER.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/READ_2016/val/Images" \
#   --output "${PATHLOG}/H091/READ_2016/images" \
#   --save_boxes_dir "${PATHLOG}/H091/READ_2016/boxes" \
#   --dataset ctw1500 \
#   --results_log "${root}/logs/ATS/Hi-SAM/stage1/Hi-SAM_gridsearch.txt" \
#   --text TrainOn_READ2016_91_cer-TestOn_READ2016 \
# >> "$output_file"

# ################# H060
# PRETRAINE_MODEL=60_2025-10-01_ID_14189314
# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/READ_2016/val/Images" \
#   --output "${PATHLOG}/H060/READ_2016/images" \
#   --save_boxes_dir "${PATHLOG}/H060/READ_2016/boxes" \
#   --dataset ctw1500 \
#   --results_log "${root}/logs/ATS/Hi-SAM/stage1/Hi-SAM_gridsearch.txt" \
#   --text TrainOn_READ2016-TestOn_READ2016 \
# >> "$output_file"


# ################# H080
# PRETRAINE_MODEL=80_2026-02-04_ID_15420249
# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/READ_2016/val/Images" \
#   --output "${PATHLOG}/H060/READ_2016/images" \
#   --save_boxes_dir "${PATHLOG}/H060/READ_2016/boxes" \
#   --dataset ctw1500 \
#   --results_log "${root}/logs/ATS/Hi-SAM/stage1/Hi-SAM_gridsearch.txt" \
#   --text TrainOn_READ2016-TestOn_READ2016 \
# >> "$output_file"

# ################# H081
# PRETRAINE_MODEL=81_2026-01-25_ID_15289657
# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/READ_2016/val/Images" \
#   --output "${PATHLOG}/H060/READ_2016/images" \
#   --save_boxes_dir "${PATHLOG}/H060/READ_2016/boxes" \
#   --dataset ctw1500 \
#   --results_log "${root}/logs/ATS/Hi-SAM/stage1/Hi-SAM_gridsearch.txt" \
#   --text TrainOn_READ2016-TestOn_READ2016 \
# >> "$output_file"

# ############################# Train on IAM #############################
# # ################# H088
# PRETRAINE_MODEL=88_2026-02-05_ID_15438914
# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/IAM/val/Images" \
#   --output "${PATHLOG}/H088/IAM/images" \
#   --save_boxes_dir "${PATHLOG}/H088/IAM/boxes" \
#   --dataset ctw1500 \
#   --results_log "${root}/logs/ATS/Hi-SAM/stage1/Hi-SAM_gridsearch.txt" \
#   --text TrainOn_IAM-TestOn_IAM \
# >> "$output_file"

# ################# H089
# PRETRAINE_MODEL=89_2026-02-05_ID_15438851
# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/IAM/val/Images" \
#   --output "${PATHLOG}/H089/IAM/images" \
#   --save_boxes_dir "${PATHLOG}/H089/IAM/boxes" \
#   --dataset ctw1500 \
#   --results_log "${root}/logs/ATS/Hi-SAM/stage1/Hi-SAM_gridsearch.txt" \
#   --text TrainOn_IAM-TestOn_IAM \
# >> "$output_file"

# ################# H090
# PRETRAINE_MODEL=90_2026-02-05_ID_15438800
# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/IAM/val/Images" \
#   --output "${PATHLOG}/H088/IAM/images" \
#   --save_boxes_dir "${PATHLOG}/H088/IAM/boxes" \
#   --dataset ctw1500 \
#   --results_log "${root}/logs/ATS/Hi-SAM/stage1/Hi-SAM_gridsearch.txt" \
#   --text TrainOn_IAM_90-TestOn_IAM \
# >> "$output_file"


# ############################ Train on NorHandv3_mini_v3 #############################
# ################ H085
# PRETRAINE_MODEL=85_2026-02-05_ID_15438860
# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/NorHandv3_mini_v3/val/Images" \
#   --output "${PATHLOG}/H085/NorHandv3_mini_v3/images" \
#   --save_boxes_dir "${PATHLOG}/H085/NorHandv3_mini_v3/boxes" \
#   --dataset ctw1500 \
#   --results_log "${root}/logs/ATS/Hi-SAM/stage1/Hi-SAM_gridsearch.txt" \
#   --text TrainOn_NorHandv3_mini_v3_85-TestOn_NorHandv3_mini_v3 \
# >> "$output_file"

# ################# H086
# PRETRAINE_MODEL=86_2026-02-05_ID_15438861
# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/NorHandv3_mini_v3/val/Images" \
#   --output "${PATHLOG}/H085/NorHandv3_mini_v3/images" \
#   --save_boxes_dir "${PATHLOG}/H085/NorHandv3_mini_v3/boxes" \
#   --dataset ctw1500 \
#   --results_log "${root}/logs/ATS/Hi-SAM/stage1/Hi-SAM_gridsearch.txt" \
#   --text TrainOn_NorHandv3_mini_v3_86-TestOn_NorHandv3_mini_v3 \
# >> "$output_file"


# ################# H087
# PRETRAINE_MODEL=87_2026-01-25_ID_15300532
# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/NorHandv3_mini_v3/val/Images" \
#   --output "${PATHLOG}/H085/NorHandv3_mini_v3/images" \
#   --save_boxes_dir "${PATHLOG}/H085/NorHandv3_mini_v3/boxes" \
#   --dataset ctw1500 \
#   --results_log "${root}/logs/ATS/Hi-SAM/stage1/Hi-SAM_gridsearch.txt" \
#   --text TrainOn_NorHandv3_mini_v3_87-TestOn_NorHandv3_mini_v3 \
# >> "$output_file"