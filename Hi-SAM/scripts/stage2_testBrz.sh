#!/bin/bash
#SBATCH -o /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.out
#SBATCH -e /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.err
#SBATCH -t 3-00:00:00
#SBATCH -C thin
#SBATCH --gpus=1


mamba init bash
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate pytorch25

# Parameters
file=demo_text_detection_mAP_stage2.py

# root=/home/x_gapat/PROJECTS
root="${HOME}/PROJECTS"
main_script="${root}/codes/Hi-SAM_Doc/${file}"


PATHLOG="${root}/logs/ATS/Hi-SAM/fewshot"
echo "path log :"
echo ${PATHLOG}

output_file="${PATHLOG}/out.txt"

export PYTHONPATH=/proj/document_analysis/users/x_gapat/codes/Hi-SAM_Doc/

PRETRAINE_MODEL=60-50_shot_2026-04-29_ID_16463213
python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATASETS/Bentham/test/Images" \
  --output "${PATHLOG}/H060_50/Bentham/images" \
  --save_boxes_dir "${PATHLOG}/H060_50/Bentham/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.55 0.5 \
  --map True \
  --img_ext jpg \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/044_2026-04-07_ID_16124600/crnn_best.torch" \
  --htr_charset "${root}/DATASETS/Bentham/box_lines_dataset/charset.txt" \
  --htr_bb_sorting IoU-matching \
  --results_log "${PATHLOG}/stage2_results_fewshot.txt" \
  --text TrainOn_READ2016_few50_100-TestOn_Bentham \
>> "$output_file"

PRETRAINE_MODEL=60-50_shot_2026-04-30_ID_16465074
python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATASETS/Bentham/test/Images" \
  --output "${PATHLOG}/H060_50/Bentham/images" \
  --save_boxes_dir "${PATHLOG}/H060_50/Bentham/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.55 0.5 \
  --map True \
  --img_ext jpg \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/044_2026-04-07_ID_16124600/crnn_best.torch" \
  --htr_charset "${root}/DATASETS/Bentham/box_lines_dataset/charset.txt" \
  --htr_bb_sorting IoU-matching \
  --results_log "${PATHLOG}/stage2_results_fewshot.txt" \
  --text TrainOn_READ2016_few50_200-TestOn_Bentham \
>> "$output_file"

PRETRAINE_MODEL=60-50_shot_2026-05-01_ID_16472770
python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATASETS/Bentham/test/Images" \
  --output "${PATHLOG}/H060_50/Bentham/images" \
  --save_boxes_dir "${PATHLOG}/H060_50/Bentham/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.55 0.5 \
  --map True \
  --img_ext jpg \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/044_2026-04-07_ID_16124600/crnn_best.torch" \
  --htr_charset "${root}/DATASETS/Bentham/box_lines_dataset/charset.txt" \
  --htr_bb_sorting IoU-matching \
  --results_log "${PATHLOG}/stage2_results_fewshot.txt" \
  --text TrainOn_READ2016_few50_200_e4-TestOn_Bentham \
>> "$output_file"

# ############################# Train on READ #############################
# ################# H060
# PRETRAINE_MODEL=60_2025-10-01_ID_14189314
# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/READ_2016/test/Images" \
#   --output "${PATHLOG}/H060/READ_2016/images" \
#   --save_boxes_dir "${PATHLOG}/H060/READ_2016/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.55 0.5 \
#   --map True \
#   --img_ext JPG \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/4_2025-12-02_ID_14885224/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/READ_2016/charset_read2016.txt" \
#   --htr_bb_sorting multi-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_READ2016-TestOn_READ2016 \
# >> "$output_file"

# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/IAM/test/Images" \
#   --output "${PATHLOG}/H060/IAM/images" \
#   --save_boxes_dir "${PATHLOG}/H060/IAM/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.55 0.5 \
#   --map True \
#   --img_ext png \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/8_2025-12-08_ID_14906240/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/IAM/charset_iam.txt" \
#   --htr_bb_sorting single-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_READ2016-TestOn_IAM \
# >> "$output_file"

# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/NorHandv3_mini_v3/test/Images" \
#   --output "${PATHLOG}/H060/NorHandv3_mini_v3/images" \
#   --save_boxes_dir "${PATHLOG}/H060/NorHandv3_mini_v3/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.55 0.5 \
#   --map True \
#   --img_ext jpg \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/40_2026-01-17_DGX_JOB_01/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/NorHandv3_mini_v3/charset_norhand_v3_mini.txt" \
#   --htr_bb_sorting multi-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_READ2016-TestOn_NorHandv3_mini_v3 \
# >> "$output_file"

# ################ H080
# PRETRAINE_MODEL=80_2026-02-04_ID_15420249
# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/READ_2016/test/Images" \
#   --output "${PATHLOG}/H080/READ_2016/images" \
#   --save_boxes_dir "${PATHLOG}/H080/READ_2016/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.8 0.45 \
#   --map True \
#   --img_ext JPG \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/14_2025-12-09_ID_14908711/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/READ_2016/charset_read2016.txt" \
#   --htr_bb_sorting multi-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_READ2016-TestOn_READ2016 \
# >> "$output_file"

# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/IAM/test/Images" \
#   --output "${PATHLOG}/H080/IAM/images" \
#   --save_boxes_dir "${PATHLOG}/H080/IAM/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.8 0.45 \
#   --map True \
#   --img_ext png \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/12_2025-12-09_ID_14908721/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/IAM/charset_iam.txt" \
#   --htr_bb_sorting single-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_READ2016-TestOn_IAM \
# >> "$output_file"

# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/NorHandv3_mini_v3/test/Images" \
#   --output "${PATHLOG}/H080/NorHandv3_mini_v3/images" \
#   --save_boxes_dir "${PATHLOG}/H080/NorHandv3_mini_v3/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.8 0.45 \
#   --map True \
#   --img_ext jpg \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/41_2026-01-17_DGX_JOB_01/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/NorHandv3_mini_v3/charset_norhand_v3_mini.txt" \
#   --htr_bb_sorting multi-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_READ2016-TestOn_NorHandv3_mini_v3 \
# >> "$output_file"

# ################ H081
# PRETRAINE_MODEL=81_2026-01-25_ID_15289657
# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/READ_2016/test/Images" \
#   --output "${PATHLOG}/H081/READ_2016/images" \
#   --save_boxes_dir "${PATHLOG}/H081/READ_2016/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.6 0.6 \
#   --map True \
#   --img_ext JPG \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/15_2025-12-09_ID_14908712/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/READ_2016/charset_read2016.txt" \
#   --htr_bb_sorting multi-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_READ2016-TestOn_READ2016 \
# >> "$output_file"

# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/IAM/test/Images" \
#   --output "${PATHLOG}/H081/IAM/images" \
#   --save_boxes_dir "${PATHLOG}/H081/IAM/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.6 0.6 \
#   --map True \
#   --img_ext png \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/13_2025-12-17_ID_14947631/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/IAM/charset_iam.txt" \
#   --htr_bb_sorting single-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_READ2016-TestOn_IAM \
# >> "$output_file"

# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/NorHandv3_mini_v3/test/Images" \
#   --output "${PATHLOG}/H081/NorHandv3_mini_v3/images" \
#   --save_boxes_dir "${PATHLOG}/H081/NorHandv3_mini_v3/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.6 0.6 \
#   --map True \
#   --img_ext jpg \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/42_2026-01-17_DGX_JOB_01/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/NorHandv3_mini_v3/charset_norhand_v3_mini.txt" \
#   --htr_bb_sorting multi-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_READ2016-TestOn_NorHandv3_mini_v3 \
# >> "$output_file"


############################# Train on IAM #############################
################# H088
# PRETRAINE_MODEL=88_2026-02-05_ID_15438914
# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/READ_2016/test/Images" \
#   --output "${PATHLOG}/H088/READ_2016/images" \
#   --save_boxes_dir "${PATHLOG}/H088/READ_2016/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.3 0.3 \
#   --map True \
#   --img_ext JPG \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/4_2025-12-02_ID_14885224/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/READ_2016/charset_read2016.txt" \
#   --htr_bb_sorting multi-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_IAM-TestOn_READ2016 \
# >> "$output_file"

# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/IAM/test/Images" \
#   --output "${PATHLOG}/H088/IAM/images" \
#   --save_boxes_dir "${PATHLOG}/H088/IAM/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.3 0.3 \
#   --map True \
#   --img_ext png \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/8_2025-12-08_ID_14906240/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/IAM/charset_iam.txt" \
#   --htr_bb_sorting single-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_IAM-TestOn_IAM \
# >> "$output_file"

# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/NorHandv3_mini_v3/test/Images" \
#   --output "${PATHLOG}/H088/NorHandv3_mini_v3/images" \
#   --save_boxes_dir "${PATHLOG}/H088/NorHandv3_mini_v3/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.3 0.3 \
#   --map True \
#   --img_ext jpg \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/40_2026-01-17_DGX_JOB_01/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/NorHandv3_mini_v3/charset_norhand_v3_mini.txt" \
#   --htr_bb_sorting multi-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_IAM-TestOn_NorHandv3_mini_v3 \
# >> "$output_file"

################# H089
# PRETRAINE_MODEL=89_2026-02-05_ID_15438851
# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/READ_2016/test/Images" \
#   --output "${PATHLOG}/H089/READ_2016/images" \
#   --save_boxes_dir "${PATHLOG}/H089/READ_2016/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.3 0.3 \
#   --map True \
#   --img_ext JPG \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/14_2025-12-09_ID_14908711/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/READ_2016/charset_read2016.txt" \
#   --htr_bb_sorting multi-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_IAM-TestOn_READ2016 \
# >> "$output_file"

# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/IAM/test/Images" \
#   --output "${PATHLOG}/H089/IAM/images" \
#   --save_boxes_dir "${PATHLOG}/H089/IAM/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.3 0.3 \
#   --map True \
#   --img_ext png \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/12_2025-12-09_ID_14908721/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/IAM/charset_iam.txt" \
#   --htr_bb_sorting single-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_IAM-TestOn_IAM \
# >> "$output_file"

# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/NorHandv3_mini_v3/test/Images" \
#   --output "${PATHLOG}/H089/NorHandv3_mini_v3/images" \
#   --save_boxes_dir "${PATHLOG}/H089/NorHandv3_mini_v3/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.3 0.3 \
#   --map True \
#   --img_ext jpg \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/41_2026-01-17_DGX_JOB_01/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/NorHandv3_mini_v3/charset_norhand_v3_mini.txt" \
#   --htr_bb_sorting multi-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_IAM-TestOn_NorHandv3_mini_v3 \
# >> "$output_file"

################# H090
# PRETRAINE_MODEL=90_2026-02-05_ID_15438800
# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/READ_2016/test/Images" \
#   --output "${PATHLOG}/H090/READ_2016/images" \
#   --save_boxes_dir "${PATHLOG}/H090/READ_2016/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.3 0.55 \
#   --map True \
#   --img_ext JPG \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/15_2025-12-09_ID_14908712/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/READ_2016/charset_read2016.txt" \
#   --htr_bb_sorting multi-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_IAM-TestOn_READ2016 \
# >> "$output_file"

# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/IAM/test/Images" \
#   --output "${PATHLOG}/H090/IAM/images" \
#   --save_boxes_dir "${PATHLOG}/H090/IAM/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.3 0.55 \
#   --map True \
#   --img_ext png \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/13_2025-12-17_ID_14947631/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/IAM/charset_iam.txt" \
#   --htr_bb_sorting single-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_IAM-TestOn_IAM \
# >> "$output_file"

# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/NorHandv3_mini_v3/test/Images" \
#   --output "${PATHLOG}/H090/NorHandv3_mini_v3/images" \
#   --save_boxes_dir "${PATHLOG}/H090/NorHandv3_mini_v3/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.3 0.55 \
#   --map True \
#   --img_ext jpg \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/42_2026-01-17_DGX_JOB_01/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/NorHandv3_mini_v3/charset_norhand_v3_mini.txt" \
#   --htr_bb_sorting multi-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_IAM-TestOn_NorHandv3_mini_v3 \
# >> "$output_file"


############################# Train on NorHandv3_mini_v3 #############################
################# H085
# PRETRAINE_MODEL=85_2026-02-05_ID_15438860
# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/READ_2016/test/Images" \
#   --output "${PATHLOG}/H085/READ_2016/images" \
#   --save_boxes_dir "${PATHLOG}/H085/READ_2016/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.65 0.65 \
#   --map True \
#   --img_ext JPG \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/4_2025-12-02_ID_14885224/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/READ_2016/charset_read2016.txt" \
#   --htr_bb_sorting multi-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_NorHandv3_mini_v3-TestOn_READ2016 \
# >> "$output_file"

# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/IAM/test/Images" \
#   --output "${PATHLOG}/H085/IAM/images" \
#   --save_boxes_dir "${PATHLOG}/H085/IAM/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.65 0.65 \
#   --map True \
#   --img_ext png \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/8_2025-12-08_ID_14906240/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/IAM/charset_iam.txt" \
#   --htr_bb_sorting single-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_NorHandv3_mini_v3-TestOn_IAM \
# >> "$output_file"

# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/NorHandv3_mini_v3/test/Images" \
#   --output "${PATHLOG}/H085/NorHandv3_mini_v3/images" \
#   --save_boxes_dir "${PATHLOG}/H085/NorHandv3_mini_v3/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.65 0.65 \
#   --map True \
#   --img_ext jpg \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/40_2026-01-17_DGX_JOB_01/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/NorHandv3_mini_v3/charset_norhand_v3_mini.txt" \
#   --htr_bb_sorting multi-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_NorHandv3_mini_v3-TestOn_NorHandv3_mini_v3 \
# >> "$output_file"

################# H086
# PRETRAINE_MODEL=86_2026-02-05_ID_15438861
# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/READ_2016/test/Images" \
#   --output "${PATHLOG}/H086/READ_2016/images" \
#   --save_boxes_dir "${PATHLOG}/H086/READ_2016/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.7 0.6 \
#   --map True \
#   --img_ext JPG \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/14_2025-12-09_ID_14908711/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/READ_2016/charset_read2016.txt" \
#   --htr_bb_sorting multi-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_NorHandv3_mini_v3-TestOn_READ2016 \
# >> "$output_file"

# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/IAM/test/Images" \
#   --output "${PATHLOG}/H086/IAM/images" \
#   --save_boxes_dir "${PATHLOG}/H086/IAM/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.7 0.6 \
#   --map True \
#   --img_ext png \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/12_2025-12-09_ID_14908721/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/IAM/charset_iam.txt" \
#   --htr_bb_sorting single-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_NorHandv3_mini_v3-TestOn_IAM \
# >> "$output_file"

# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/NorHandv3_mini_v3/test/Images" \
#   --output "${PATHLOG}/H086/NorHandv3_mini_v3/images" \
#   --save_boxes_dir "${PATHLOG}/H086/NorHandv3_mini_v3/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.7 0.6 \
#   --map True \
#   --img_ext jpg \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/41_2026-01-17_DGX_JOB_01/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/NorHandv3_mini_v3/charset_norhand_v3_mini.txt" \
#   --htr_bb_sorting multi-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_NorHandv3_mini_v3-TestOn_NorHandv3_mini_v3 \
# >> "$output_file"

################# H087
# PRETRAINE_MODEL=87_2026-01-25_ID_15300532
# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/READ_2016/test/Images" \
#   --output "${PATHLOG}/H087/READ_2016/images" \
#   --save_boxes_dir "${PATHLOG}/H087/READ_2016/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.7 0.5 \
#   --map True \
#   --img_ext JPG \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/15_2025-12-09_ID_14908712/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/READ_2016/charset_read2016.txt" \
#   --htr_bb_sorting multi-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_NorHandv3_mini_v3-TestOn_READ2016 \
# >> "$output_file"

# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/IAM/test/Images" \
#   --output "${PATHLOG}/H087/IAM/images" \
#   --save_boxes_dir "${PATHLOG}/H087/IAM/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.7 0.5 \
#   --map True \
#   --img_ext png \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/13_2025-12-17_ID_14947631/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/IAM/charset_iam.txt" \
#   --htr_bb_sorting single-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_NorHandv3_mini_v3-TestOn_IAM \
# >> "$output_file"

# python "$main_script" \
#   --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
#   --model-type vit_h \
#   --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
#   --input "${root}/DATASETS/NorHandv3_mini_v3/test/Images" \
#   --output "${PATHLOG}/H087/NorHandv3_mini_v3/images" \
#   --save_boxes_dir "${PATHLOG}/H087/NorHandv3_mini_v3/boxes" \
#   --dataset ctw1500 \
#   --save_boxes True \
#   --exp_id 0 \
#   --nms 0.7 0.5 \
#   --map True \
#   --img_ext jpg \
#   --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/42_2026-01-17_DGX_JOB_01/crnn_best.torch" \
#   --htr_charset "${root}/DATASETS/NorHandv3_mini_v3/charset_norhand_v3_mini.txt" \
#   --htr_bb_sorting multi-col-sorting \
#   --results_log "${PATHLOG}/stage2_results.txt" \
#   --text TrainOn_NorHandv3_mini_v3-TestOn_NorHandv3_mini_v3 \
# >> "$output_file"