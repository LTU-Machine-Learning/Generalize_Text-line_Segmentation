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
# module load Mambaforge/23.3.1-1-hpc1-bdist
# mamba activate pytorch25

# Parameters
file=demo_text_detection_mAP.py

#root=/home/x_gapat/PROJECTS
#data_root=/proj/document_analysis/users/shared/
root=/home/gayapath/PROJECTS
data_root=/home/gayapath/PROJECTS/DATASETS

main_script="${root}/codes/Hi-SAM_Doc/${file}"

PATHLOG="${root}/logs/Hi-SAM_Doc/"
output_file="${PATHLOG}/results.txt"
output_file2="${PATHLOG}/out.txt"

PRETRAINE_PATH="${PATHLOG}/pretrained_checkpoint"


DATA_IN="${data_root}/READ_2016/Test/Images"
DATA_OUT="${root}/logs/Hi-SAM_Doc/sample_output"

export PYTHONPATH=/proj/document_analysis/users/x_gapat/codes/Hi-SAM_Doc/


# The job

CUDA_VISIBLE_DEVICES=3 python "$main_script" \
  --checkpoint "${PATHLOG}/57_2025-09-25_ID_/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "$PRETRAINE_PATH" \
  --input /home/gayapath/PROJECTS/DATA_DGX2/READ_2016/Test/Images \
  --output /home/gayapath/PROJECTS/logs/Hi-SAM_Doc/sample_output/H057/READ_2016/images \
  --save_boxes_dir /home/gayapath/PROJECTS/logs/Hi-SAM_Doc/sample_output/H057/READ_2016/boxes \
  --dataset ctw1500 \
  --nms 0.6 0.6 \
  --results_log results_Testset_H057.txt \
  --save_boxes True \
  --map True \
  --text READ_2016 \
>> "$output_file2"

CUDA_VI
SIBLE_DEVICES=3 python "$main_script" \
  --checkpoint "${PATHLOG}/57_2025-09-25_ID_/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "$PRETRAINE_PATH" \
  --input /home/gayapath/PROJECTS/DATA_DGX2/IAM/Page/Test/Images \
  --output /home/gayapath/PROJECTS/logs/Hi-SAM_Doc/sample_output/H057/IAM/Images \
  --save_boxes_dir /home/gayapath/PROJECTS/logs/Hi-SAM_Doc/sample_output/H057/IAM/boxes \
  --dataset ctw1500 \
  --nms 0.6 0.6 \
  --results_log results_Testset_H057.txt \
  --save_boxes True \
  --map True \
  --text IAM \
>> "$output_file2"


CUDA_VISIBLE_DEVICES=3 python "$main_script" \
  --checkpoint "${PATHLOG}/57_2025-09-25_ID_/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "$PRETRAINE_PATH" \
  --input /home/gayapath/PROJECTS/DATA_DGX2/NorHandv3/test/images \
  --output /home/gayapath/PROJECTS/logs/Hi-SAM_Doc/sample_output/H057/NorHand/images \
  --save_boxes_dir /home/gayapath/PROJECTS/logs/Hi-SAM_Doc/sample_output/H057/NorHand/boxes \
  --dataset ctw1500 \
  --nms 0.6 0.6 \
  --results_log results_Testset_H057.txt \
  --save_boxes True \
  --map True \
  --text NorHand \
>> "$output_file2"
