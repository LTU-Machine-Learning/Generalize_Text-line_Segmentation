#!/bin/bash
#SBATCH -o /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.out
#SBATCH -e /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.err
#SBATCH -t 3-00:00:00
#SBATCH -C thin
#SBATCH --gpus=1


# mamba init bash
# module load Mambaforge/23.3.1-1-hpc1-bdist
# mamba activate pytorch25

# Parameters
file=demo_text_detection_mAP_stage3.py

# root=/home/x_gapat/PROJECTS
root=/home/gayapath/PROJECTS
main_script="${root}/codes/Hi-SAM_Doc/${file}"


PATHLOG="${root}/logs/Hi-SAM_Doc/sample_output3"
echo "path log :"
echo ${PATHLOG}

output_file="${PATHLOG}/out.txt"

export PYTHONPATH=/proj/document_analysis/users/x_gapat/codes/Hi-SAM_Doc/

############################# Train on READ #############################
################# H091
PRETRAINE_MODEL=91_2026-02-05_ID_15442927
CUDA_VISIBLE_DEVICES=7 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_CER.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATASETS/READ_2016/test/Images" \
  --output "${PATHLOG}/H091/READ_2016/images" \
  --save_boxes_dir "${PATHLOG}/H091/READ_2016/boxes" \
  --save_boxes True \
  --save_visualization True \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.6 0.6 \
  --map True \
  --img_ext JPG \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/4_2025-12-02_ID_14885224/crnn_best.torch" \
  --htr_charset "${root}/DATASETS/READ_2016/charset_read2016.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage3_results_1.txt" \
  --text TrainOn_READ2016-TestOn_READ2016 \
>> "$output_file"

CUDA_VISIBLE_DEVICES=7 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_CER.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATASETS/IAM/pages_cleaned/test/Images" \
  --output "${PATHLOG}/H091/IAM/images" \
  --save_boxes_dir "${PATHLOG}/H091/IAM/boxes" \
  --save_boxes True \
  --save_visualization True \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.6 0.6 \
  --map True \
  --img_ext png \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/8_2025-12-08_ID_14906240/crnn_best.torch" \
  --htr_charset "${root}/DATASETS/IAM/pages_cleaned/charset_iam.txt" \
  --htr_bb_sorting single-col-sorting \
  --results_log "${PATHLOG}/stage3_results_1.txt" \
  --text TrainOn_READ2016-TestOn_IAM \
>> "$output_file"

CUDA_VISIBLE_DEVICES=7 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_CER.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATASETS/NorHandv3_mini_v3/test/Images" \
  --output "${PATHLOG}/H091/NorHandv3_mini_v3/images" \
  --save_boxes_dir "${PATHLOG}/H091/NorHandv3_mini_v3/boxes" \
  --save_boxes True \
  --save_visualization True \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.6 0.6 \
  --map True \
  --img_ext jpg \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/40_2026-01-17_DGX_JOB_01/crnn_best.torch" \
  --htr_charset "${root}/DATASETS/NorHandv3_mini_v3/charset_norhand_v3_mini.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage3_results_1.txt" \
  --text TrainOn_READ2016-TestOn_NorHandv3_mini_v3 \
>> "$output_file"

################# H092
PRETRAINE_MODEL=92_2026-02-09_ID_15506966
CUDA_VISIBLE_DEVICES=7 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_CER.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATASETS/READ_2016/test/Images" \
  --output "${PATHLOG}/H092/READ_2016/images" \
  --save_boxes_dir "${PATHLOG}/H092/READ_2016/boxes" \
  --save_boxes True \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.6 0.6 \
  --map True \
  --img_ext JPG \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/14_2025-12-09_ID_14908711/crnn_best.torch" \
  --htr_charset "${root}/DATASETS/READ_2016/charset_read2016.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage3_results_1.txt" \
  --text TrainOn_READ2016-TestOn_READ2016 \
>> "$output_file"

CUDA_VISIBLE_DEVICES=7 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_CER.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATASETS/IAM/pages_cleaned/test/Images" \
  --output "${PATHLOG}/H092/IAM/images" \
  --save_boxes_dir "${PATHLOG}/H092/IAM/boxes" \
  --save_boxes True \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.6 0.6 \
  --map True \
  --img_ext png \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/12_2025-12-09_ID_14908721/crnn_best.torch" \
  --htr_charset "${root}/DATASETS/IAM/pages_cleaned/charset_iam.txt" \
  --htr_bb_sorting single-col-sorting \
  --results_log "${PATHLOG}/stage3_results_1.txt" \
  --text TrainOn_READ2016-TestOn_IAM \
>> "$output_file"

CUDA_VISIBLE_DEVICES=7 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_CER.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATASETS/NorHandv3_mini_v3/test/Images" \
  --output "${PATHLOG}/H092/NorHandv3_mini_v3/images" \
  --save_boxes_dir "${PATHLOG}/H092/NorHandv3_mini_v3/boxes" \
  --save_boxes True \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.6 0.6 \
  --map True \
  --img_ext jpg \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/41_2026-01-17_DGX_JOB_01/crnn_best.torch" \
  --htr_charset "${root}/DATASETS/NorHandv3_mini_v3/charset_norhand_v3_mini.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage3_results_1.txt" \
  --text TrainOn_READ2016-TestOn_NorHandv3_mini_v3 \
>> "$output_file"

################# H093
PRETRAINE_MODEL=93_2026-02-09_ID_15506977
CUDA_VISIBLE_DEVICES=7 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_CER.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATASETS/READ_2016/test/Images" \
  --output "${PATHLOG}/H093/READ_2016/images" \
  --save_boxes_dir "${PATHLOG}/H093/READ_2016/boxes" \
  --save_boxes True \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.6 0.6 \
  --map True \
  --img_ext JPG \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/15_2025-12-09_ID_14908712/crnn_best.torch" \
  --htr_charset "${root}/DATASETS/READ_2016/charset_read2016.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage3_results_1.txt" \
  --text TrainOn_READ2016-TestOn_READ2016 \
>> "$output_file"

CUDA_VISIBLE_DEVICES=7 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_CER.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATASETS/IAM/pages_cleaned/test/Images" \
  --output "${PATHLOG}/H093/IAM/images" \
  --save_boxes_dir "${PATHLOG}/H093/IAM/boxes" \
  --save_boxes True \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.6 0.6 \
  --map True \
  --img_ext png \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/13_2025-12-17_ID_14947631/crnn_best.torch" \
  --htr_charset "${root}/DATASETS/IAM/pages_cleaned/charset_iam.txt" \
  --htr_bb_sorting single-col-sorting \
  --results_log "${PATHLOG}/stage3_results_1.txt" \
  --text TrainOn_READ2016-TestOn_IAM \
>> "$output_file"

CUDA_VISIBLE_DEVICES=7 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_CER.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATASETS/NorHandv3_mini_v3/test/Images" \
  --output "${PATHLOG}/H093/NorHandv3_mini_v3/images" \
  --save_boxes_dir "${PATHLOG}/H093/NorHandv3_mini_v3/boxes" \
  --save_boxes True \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.6 0.6 \
  --map True \
  --img_ext jpg \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/42_2026-01-17_DGX_JOB_01/crnn_best.torch" \
  --htr_charset "${root}/DATASETS/NorHandv3_mini_v3/charset_norhand_v3_mini.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage3_results_1.txt" \
  --text TrainOn_READ2016-TestOn_NorHandv3_mini_v3 \
>> "$output_file"


