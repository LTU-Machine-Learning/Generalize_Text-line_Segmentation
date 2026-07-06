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
file=demo_text_detection_mAP.py

root=/home/gayapath/PROJECTS
data_root=/home/gayapath/PROJECTS/DATASETS


main_script="${root}/codes/Hi-SAM_Doc/${file}"

PATHLOG="${root}/logs/Hi-SAM_Doc/"
output_file="${PATHLOG}/results_stage1_all_models_dgx.txt"
output_file2="${PATHLOG}/out.txt"

PRETRAINE_PATH="${PATHLOG}/pretrained_checkpoint"


DATA_IN_READ2016="${data_root}/READ_2016/Test/Images"
DATA_IN_IAM="${data_root}/IAM/Page/test/images"
DATA_IN_Norhandv3="${data_root}/NorHandv3/test/cleaned_images"
DATA_OUT="${root}/logs/Hi-SAM_Doc/output_images"

export PYTHONPATH=/proj/document_analysis/users/x_gapat/codes/Hi-SAM_Doc/


EXP="57_2025-09-25_ID_"
CHECKPOINT="${PATHLOG}/${EXP}/saved_model"
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
  --results_log "$output_file" \
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
  --results_log "$output_file" \
>> "$output_file2"

# CUDA_VISIBLE_DEVICES=3 python "$main_script" \
#   --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "$PRETRAINE_PATH" \
#   --input "$DATA_IN_Norhandv3" \
#   --output "$DATA_OUT" \
#   --dataset ctw1500 \
#   --map True \
#   --nms 0.6 0.6 \
#   --text Norhandv3 \
#   --results_log "$output_file" \
# >> "$output_file2"

#####################################################################
EXP="77_2025-12-09_ID_14908887"
CHECKPOINT="${PATHLOG}/${EXP}/saved_model"
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
  --results_log "$output_file" \
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
  --results_log "$output_file" \
>> "$output_file2"

# CUDA_VISIBLE_DEVICES=3 python "$main_script" \
#   --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "$PRETRAINE_PATH" \
#   --input "$DATA_IN_Norhandv3" \
#   --output "$DATA_OUT" \
#   --dataset ctw1500 \
#   --map True \
#   --nms 0.6 0.6 \
#   --text Norhandv3 \
#   --results_log "$output_file" \
# >> "$output_file2"

#####################################################################
EXP="78_2025-12-09_ID_14909192"
CHECKPOINT="${PATHLOG}/${EXP}/saved_model"
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
  --results_log "$output_file" \
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
  --results_log "$output_file" \
>> "$output_file2"

# CUDA_VISIBLE_DEVICES=3 python "$main_script" \
#   --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "$PRETRAINE_PATH" \
#   --input "$DATA_IN_Norhandv3" \
#   --output "$DATA_OUT" \
#   --dataset ctw1500 \
#   --map True \
#   --nms 0.6 0.6 \
#   --text Norhandv3 \
#   --results_log "$output_file" \
# >> "$output_file2"

#####################################################################
EXP="79_2025-12-09_ID_14909194"
CHECKPOINT="${PATHLOG}/${EXP}/saved_model"
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
  --results_log "$output_file" \
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
  --results_log "$output_file" \
>> "$output_file2"

# CUDA_VISIBLE_DEVICES=3 python "$main_script" \
#   --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "$PRETRAINE_PATH" \
#   --input "$DATA_IN_Norhandv3" \
#   --output "$DATA_OUT" \
#   --dataset ctw1500 \
#   --map True \
#   --nms 0.6 0.6 \
#   --text Norhandv3 \
#   --results_log "$output_file" \
# >> "$output_file2"



#####################################################################
EXP="60_2025-10-01_ID_14189314"
CHECKPOINT="${PATHLOG}/${EXP}/saved_model"
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
  --results_log "$output_file" \
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
  --results_log "$output_file" \
>> "$output_file2"

# CUDA_VISIBLE_DEVICES=3 python "$main_script" \
#   --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "$PRETRAINE_PATH" \
#   --input "$DATA_IN_Norhandv3" \
#   --output "$DATA_OUT" \
#   --dataset ctw1500 \
#   --map True \
#   --nms 0.6 0.6 \
#   --text Norhandv3 \
#   --results_log "$output_file" \
# >> "$output_file2"

#####################################################################
EXP="80_2025-12-09_ID_14909195"
CHECKPOINT="${PATHLOG}/${EXP}/saved_model"
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
  --results_log "$output_file" \
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
  --results_log "$output_file" \
>> "$output_file2"

# CUDA_VISIBLE_DEVICES=3 python "$main_script" \
#   --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "$PRETRAINE_PATH" \
#   --input "$DATA_IN_Norhandv3" \
#   --output "$DATA_OUT" \
#   --dataset ctw1500 \
#   --map True \
#   --nms 0.6 0.6 \
#   --text Norhandv3 \
#   --results_log "$output_file" \
# >> "$output_file2"

#####################################################################
EXP="81_2025-12-09_ID_14909197"
CHECKPOINT="${PATHLOG}/${EXP}/saved_model"
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
  --results_log "$output_file" \
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
  --results_log "$output_file" \
>> "$output_file2"

# CUDA_VISIBLE_DEVICES=3 python "$main_script" \
#   --checkpoint "${CHECKPOINT}/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "$PRETRAINE_PATH" \
#   --input "$DATA_IN_Norhandv3" \
#   --output "$DATA_OUT" \
#   --dataset ctw1500 \
#   --map True \
#   --nms 0.6 0.6 \
#   --text Norhandv3 \
#   --results_log "$output_file" \
# >> "$output_file2"
