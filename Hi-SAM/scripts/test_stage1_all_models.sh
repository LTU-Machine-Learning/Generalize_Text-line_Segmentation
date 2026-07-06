#!/bin/bash
#SBATCH -o /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.out
#SBATCH -e /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.err
#SBATCH -t 0-02:00:00
#SBATCH -C thin
#SBATCH --gpus=1


# mamba init bash
# module load Mambaforge/23.3.1-1-hpc1-bdist
# mamba activate pytorch25

# Parameters
file=demo_text_detection_mAP_stage1.py

# root=/home/x_gapat/PROJECTS
# data_root=/proj/document_analysis/users/x_gapat/DATASETS
root=/home/gayapath/PROJECTS
data_root=/home/gayapath/PROJECTS/DATA_DGX2


main_script="${root}/codes/Hi-SAM_Doc/${file}"

PATHLOG="${root}/logs/ATS/Hi-SAM/Pred_new"
output_file="${PATHLOG}/results_stage1.txt"
output_file2="${PATHLOG}/out_st1.txt"

PRETRAINE_PATH="${root}/logs/Hi-SAM_Doc/pretrained_checkpoint"


DATA_IN_READ2016="${data_root}/READ_2016/test/Images"
DATA_IN_IAM="${data_root}/IAM/pages_cleaned/test/Images"
DATA_IN_Norhandv3="${data_root}/NorHandv3_mini_v3/test/Images"
DATA_OUT="${root}/logs/Hi-SAM_Doc/output_images"

export PYTHONPATH=/proj/document_analysis/users/x_gapat/codes/Hi-SAM_Doc/


#####################################################################
EXP="60_2025-10-01_ID_14189314"
CHECKPOINT="${root}/logs/Hi-SAM_Doc/${EXP}/saved_model"
CUDA_VISIBLE_DEVICES=3 python "$main_script" \
  --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "$PRETRAINE_PATH" \
  --input "$DATA_IN_READ2016" \
  --output "$DATA_OUT" \
  --dataset ctw1500 \
  --map True \
  --nms 0.55 0.5 \
  --text READ_2016 \
  --results_log "${PATHLOG}/Hi-SAM_READ 2016.txt" \
>> "$output_file2"

CUDA_VISIBLE_DEVICES=3 python "$main_script" \
  --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "$PRETRAINE_PATH" \
  --input "$DATA_IN_IAM" \
  --output "$DATA_OUT" \
  --dataset ctw1500 \
  --map True \
  --nms 0.55 0.5 \
  --text IAM \
  --results_log "${PATHLOG}/Hi-SAM_IAM.txt" \
>> "$output_file2"

CUDA_VISIBLE_DEVICES=3 python "$main_script" \
  --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "$PRETRAINE_PATH" \
  --input "$DATA_IN_Norhandv3" \
  --output "$DATA_OUT" \
  --dataset ctw1500 \
  --map True \
  --nms 0.55 0.5 \
  --text NorHandv3_mini_v3 \
  --results_log "${PATHLOG}/Hi-SAM_NorHandV3.txt" \
>> "$output_file2"

#####################################################################
EXP="80_2025-12-09_ID_14909195"
CHECKPOINT="${root}/logs/Hi-SAM_Doc/${EXP}/saved_model"
CUDA_VISIBLE_DEVICES=3 python "$main_script" \
  --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "$PRETRAINE_PATH" \
  --input "$DATA_IN_READ2016" \
  --output "$DATA_OUT" \
  --dataset ctw1500 \
  --map True \
  --nms 0.8 0.45 \
  --text READ_2016 \
  --results_log "${PATHLOG}/Hi-SAM_READ 2016.txt" \
>> "$output_file2"

CUDA_VISIBLE_DEVICES=3 python "$main_script" \
  --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "$PRETRAINE_PATH" \
  --input "$DATA_IN_IAM" \
  --output "$DATA_OUT" \
  --dataset ctw1500 \
  --map True \
  --nms 0.8 0.45 \
  --text IAM \
  --results_log "${PATHLOG}/Hi-SAM_IAM.txt" \
>> "$output_file2"

CUDA_VISIBLE_DEVICES=3 python "$main_script" \
  --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "$PRETRAINE_PATH" \
  --input "$DATA_IN_Norhandv3" \
  --output "$DATA_OUT" \
  --dataset ctw1500 \
  --map True \
  --nms 0.8 0.45 \
  --text NorHandv3_mini_v3 \
  --results_log "${PATHLOG}/Hi-SAM_NorHandV3.txt" \
>> "$output_file2"

#####################################################################
EXP="81_2025-12-09_ID_14909197"
CHECKPOINT="${root}/logs/Hi-SAM_Doc/${EXP}/saved_model"
CUDA_VISIBLE_DEVICES=3 python "$main_script" \
  --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "$PRETRAINE_PATH" \
  --input "$DATA_IN_READ2016" \
  --output "$DATA_OUT" \
  --dataset ctw1500 \
  --map True \
  --nms 0.6 0.6 \
  --text READ_2016 \
  --results_log "${PATHLOG}/Hi-SAM_READ 2016.txt" \
>> "$output_file2"

CUDA_VISIBLE_DEVICES=3 python "$main_script" \
  --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "$PRETRAINE_PATH" \
  --input "$DATA_IN_IAM" \
  --output "$DATA_OUT" \
  --dataset ctw1500 \
  --map True \
  --nms 0.6 0.6 \
  --text IAM \
  --results_log "${PATHLOG}/Hi-SAM_IAM.txt" \
>> "$output_file2"

CUDA_VISIBLE_DEVICES=3 python "$main_script" \
  --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "$PRETRAINE_PATH" \
  --input "$DATA_IN_Norhandv3" \
  --output "$DATA_OUT" \
  --dataset ctw1500 \
  --map True \
  --nms 0.6 0.6 \
  --text NorHandv3_mini_v3 \
  --results_log "${PATHLOG}/Hi-SAM_NorHandV3.txt" \
>> "$output_file2"
