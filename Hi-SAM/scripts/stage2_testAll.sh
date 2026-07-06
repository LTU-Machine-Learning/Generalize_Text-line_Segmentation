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
file=demo_text_detection_mAP_stage2.py

# root=/home/x_gapat/PROJECTS
root=/home/gayapath/PROJECTS
main_script="${root}/codes/Hi-SAM_Doc/${file}"


PATHLOG="${root}/logs/ATS/Hi-SAM/Pred_new_bayesmAP"
echo "path log :"
echo ${PATHLOG}

output_file="${PATHLOG}/out.txt"

export PYTHONPATH=/proj/document_analysis/users/x_gapat/codes/Hi-SAM_Doc/

############################# Train on READ #############################
################# H060
PRETRAINE_MODEL=60_2025-10-01_ID_14189314
CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/READ_2016/test/Images" \
  --output "${PATHLOG}/H060/READ_2016/images" \
  --save_boxes_dir "${PATHLOG}/H060/READ_2016/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.55 0.5 \
  --map True \
  --img_ext JPG \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/4_2025-12-02_ID_14885224/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/READ_2016/charset_read2016.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_READ2016-TestOn_READ2016 \
>> "$output_file"

CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/IAM/pages_cleaned/test/Images" \
  --output "${PATHLOG}/H060/IAM/images" \
  --save_boxes_dir "${PATHLOG}/H060/IAM/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.55 0.5 \
  --map True \
  --img_ext png \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/8_2025-12-08_ID_14906240/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/IAM/pages_cleaned/charset_iam.txt" \
  --htr_bb_sorting single-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_READ2016-TestOn_IAM \
>> "$output_file"

CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/NorHandv3_mini_v3/test/Images" \
  --output "${PATHLOG}/H060/NorHandv3_mini_v3/images" \
  --save_boxes_dir "${PATHLOG}/H060/NorHandv3_mini_v3/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.55 0.5 \
  --map True \
  --img_ext jpg \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/40_2026-01-17_DGX_JOB_01/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/NorHandv3_mini_v3/charset_norhand_v3_mini.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_READ2016-TestOn_NorHandv3_mini_v3 \
>> "$output_file"

################# H080
PRETRAINE_MODEL=80_2026-02-04_ID_15420249
CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/READ_2016/test/Images" \
  --output "${PATHLOG}/H080/READ_2016/images" \
  --save_boxes_dir "${PATHLOG}/H080/READ_2016/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.8 0.45 \
  --map True \
  --img_ext JPG \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/14_2025-12-09_ID_14908711/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/READ_2016/charset_read2016.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_READ2016-TestOn_READ2016 \
>> "$output_file"

CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/IAM/pages_cleaned/test/Images" \
  --output "${PATHLOG}/H080/IAM/images" \
  --save_boxes_dir "${PATHLOG}/H080/IAM/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.8 0.45 \
  --map True \
  --img_ext png \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/12_2025-12-09_ID_14908721/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/IAM/pages_cleaned/charset_iam.txt" \
  --htr_bb_sorting single-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_READ2016-TestOn_IAM \
>> "$output_file"

CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/NorHandv3_mini_v3/test/Images" \
  --output "${PATHLOG}/H080/NorHandv3_mini_v3/images" \
  --save_boxes_dir "${PATHLOG}/H080/NorHandv3_mini_v3/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.8 0.45 \
  --map True \
  --img_ext jpg \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/41_2026-01-17_DGX_JOB_01/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/NorHandv3_mini_v3/charset_norhand_v3_mini.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_READ2016-TestOn_NorHandv3_mini_v3 \
>> "$output_file"

################# H081
PRETRAINE_MODEL=81_2026-01-25_ID_15289657
CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/READ_2016/test/Images" \
  --output "${PATHLOG}/H081/READ_2016/images" \
  --save_boxes_dir "${PATHLOG}/H081/READ_2016/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.6 0.6 \
  --map True \
  --img_ext JPG \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/15_2025-12-09_ID_14908712/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/READ_2016/charset_read2016.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_READ2016-TestOn_READ2016 \
>> "$output_file"

CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/IAM/pages_cleaned/test/Images" \
  --output "${PATHLOG}/H081/IAM/images" \
  --save_boxes_dir "${PATHLOG}/H081/IAM/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.6 0.6 \
  --map True \
  --img_ext png \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/13_2025-12-17_ID_14947631/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/IAM/pages_cleaned/charset_iam.txt" \
  --htr_bb_sorting single-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_READ2016-TestOn_IAM \
>> "$output_file"

CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/NorHandv3_mini_v3/test/Images" \
  --output "${PATHLOG}/H081/NorHandv3_mini_v3/images" \
  --save_boxes_dir "${PATHLOG}/H081/NorHandv3_mini_v3/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.6 0.6 \
  --map True \
  --img_ext jpg \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/42_2026-01-17_DGX_JOB_01/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/NorHandv3_mini_v3/charset_norhand_v3_mini.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_READ2016-TestOn_NorHandv3_mini_v3 \
>> "$output_file"


############################# Train on IAM #############################
################# H088
PRETRAINE_MODEL=88_2026-02-05_ID_15438914
CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/READ_2016/test/Images" \
  --output "${PATHLOG}/H088/READ_2016/images" \
  --save_boxes_dir "${PATHLOG}/H088/READ_2016/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.3 0.3 \
  --map True \
  --img_ext JPG \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/4_2025-12-02_ID_14885224/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/READ_2016/charset_read2016.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_IAM-TestOn_READ2016 \
>> "$output_file"

CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/IAM/pages_cleaned/test/Images" \
  --output "${PATHLOG}/H088/IAM/images" \
  --save_boxes_dir "${PATHLOG}/H088/IAM/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.3 0.3 \
  --map True \
  --img_ext png \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/8_2025-12-08_ID_14906240/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/IAM/pages_cleaned/charset_iam.txt" \
  --htr_bb_sorting single-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_IAM-TestOn_IAM \
>> "$output_file"

CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/NorHandv3_mini_v3/test/Images" \
  --output "${PATHLOG}/H088/NorHandv3_mini_v3/images" \
  --save_boxes_dir "${PATHLOG}/H088/NorHandv3_mini_v3/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.3 0.3 \
  --map True \
  --img_ext jpg \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/40_2026-01-17_DGX_JOB_01/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/NorHandv3_mini_v3/charset_norhand_v3_mini.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_IAM-TestOn_NorHandv3_mini_v3 \
>> "$output_file"

################# H089
PRETRAINE_MODEL=89_2026-02-05_ID_15438851
CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/READ_2016/test/Images" \
  --output "${PATHLOG}/H089/READ_2016/images" \
  --save_boxes_dir "${PATHLOG}/H089/READ_2016/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.3 0.3 \
  --map True \
  --img_ext JPG \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/14_2025-12-09_ID_14908711/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/READ_2016/charset_read2016.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_IAM-TestOn_READ2016 \
>> "$output_file"

CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/IAM/pages_cleaned/test/Images" \
  --output "${PATHLOG}/H089/IAM/images" \
  --save_boxes_dir "${PATHLOG}/H089/IAM/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.3 0.3 \
  --map True \
  --img_ext png \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/12_2025-12-09_ID_14908721/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/IAM/pages_cleaned/charset_iam.txt" \
  --htr_bb_sorting single-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_IAM-TestOn_IAM \
>> "$output_file"

CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/NorHandv3_mini_v3/test/Images" \
  --output "${PATHLOG}/H089/NorHandv3_mini_v3/images" \
  --save_boxes_dir "${PATHLOG}/H089/NorHandv3_mini_v3/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.3 0.3 \
  --map True \
  --img_ext jpg \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/41_2026-01-17_DGX_JOB_01/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/NorHandv3_mini_v3/charset_norhand_v3_mini.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_IAM-TestOn_NorHandv3_mini_v3 \
>> "$output_file"

################# H090
PRETRAINE_MODEL=90_2026-02-05_ID_15438800
CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/READ_2016/test/Images" \
  --output "${PATHLOG}/H090/READ_2016/images" \
  --save_boxes_dir "${PATHLOG}/H090/READ_2016/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.3 0.55 \
  --map True \
  --img_ext JPG \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/15_2025-12-09_ID_14908712/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/READ_2016/charset_read2016.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_IAM-TestOn_READ2016 \
>> "$output_file"

CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/IAM/pages_cleaned/test/Images" \
  --output "${PATHLOG}/H090/IAM/images" \
  --save_boxes_dir "${PATHLOG}/H090/IAM/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.3 0.55 \
  --map True \
  --img_ext png \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/13_2025-12-17_ID_14947631/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/IAM/pages_cleaned/charset_iam.txt" \
  --htr_bb_sorting single-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_IAM-TestOn_IAM \
>> "$output_file"

CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/NorHandv3_mini_v3/test/Images" \
  --output "${PATHLOG}/H090/NorHandv3_mini_v3/images" \
  --save_boxes_dir "${PATHLOG}/H090/NorHandv3_mini_v3/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.3 0.55 \
  --map True \
  --img_ext jpg \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/42_2026-01-17_DGX_JOB_01/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/NorHandv3_mini_v3/charset_norhand_v3_mini.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_IAM-TestOn_NorHandv3_mini_v3 \
>> "$output_file"


############################# Train on NorHandv3_mini_v3 #############################
################# H085
PRETRAINE_MODEL=85_2026-02-05_ID_15438860
CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/READ_2016/test/Images" \
  --output "${PATHLOG}/H085/READ_2016/images" \
  --save_boxes_dir "${PATHLOG}/H085/READ_2016/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.65 0.65 \
  --map True \
  --img_ext JPG \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/4_2025-12-02_ID_14885224/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/READ_2016/charset_read2016.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_NorHandv3_mini_v3-TestOn_READ2016 \
>> "$output_file"

CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/IAM/pages_cleaned/test/Images" \
  --output "${PATHLOG}/H085/IAM/images" \
  --save_boxes_dir "${PATHLOG}/H085/IAM/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.65 0.65 \
  --map True \
  --img_ext png \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/8_2025-12-08_ID_14906240/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/IAM/pages_cleaned/charset_iam.txt" \
  --htr_bb_sorting single-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_NorHandv3_mini_v3-TestOn_IAM \
>> "$output_file"

CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/NorHandv3_mini_v3/test/Images" \
  --output "${PATHLOG}/H085/NorHandv3_mini_v3/images" \
  --save_boxes_dir "${PATHLOG}/H085/NorHandv3_mini_v3/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.65 0.65 \
  --map True \
  --img_ext jpg \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/40_2026-01-17_DGX_JOB_01/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/NorHandv3_mini_v3/charset_norhand_v3_mini.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_NorHandv3_mini_v3-TestOn_NorHandv3_mini_v3 \
>> "$output_file"

################# H086
PRETRAINE_MODEL=86_2026-02-05_ID_15438861
CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/READ_2016/test/Images" \
  --output "${PATHLOG}/H086/READ_2016/images" \
  --save_boxes_dir "${PATHLOG}/H086/READ_2016/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.7 0.6 \
  --map True \
  --img_ext JPG \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/14_2025-12-09_ID_14908711/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/READ_2016/charset_read2016.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_NorHandv3_mini_v3-TestOn_READ2016 \
>> "$output_file"

CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/IAM/pages_cleaned/test/Images" \
  --output "${PATHLOG}/H086/IAM/images" \
  --save_boxes_dir "${PATHLOG}/H086/IAM/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.7 0.6 \
  --map True \
  --img_ext png \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/12_2025-12-09_ID_14908721/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/IAM/pages_cleaned/charset_iam.txt" \
  --htr_bb_sorting single-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_NorHandv3_mini_v3-TestOn_IAM \
>> "$output_file"

CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/NorHandv3_mini_v3/test/Images" \
  --output "${PATHLOG}/H086/NorHandv3_mini_v3/images" \
  --save_boxes_dir "${PATHLOG}/H086/NorHandv3_mini_v3/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.7 0.6 \
  --map True \
  --img_ext jpg \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/41_2026-01-17_DGX_JOB_01/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/NorHandv3_mini_v3/charset_norhand_v3_mini.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_NorHandv3_mini_v3-TestOn_NorHandv3_mini_v3 \
>> "$output_file"

################# H087
PRETRAINE_MODEL=87_2026-01-25_ID_15300532
CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/READ_2016/test/Images" \
  --output "${PATHLOG}/H087/READ_2016/images" \
  --save_boxes_dir "${PATHLOG}/H087/READ_2016/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.7 0.5 \
  --map True \
  --img_ext JPG \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/15_2025-12-09_ID_14908712/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/READ_2016/charset_read2016.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_NorHandv3_mini_v3-TestOn_READ2016 \
>> "$output_file"

CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/IAM/pages_cleaned/test/Images" \
  --output "${PATHLOG}/H087/IAM/images" \
  --save_boxes_dir "${PATHLOG}/H087/IAM/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.7 0.5 \
  --map True \
  --img_ext png \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/13_2025-12-17_ID_14947631/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/IAM/pages_cleaned/charset_iam.txt" \
  --htr_bb_sorting single-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_NorHandv3_mini_v3-TestOn_IAM \
>> "$output_file"

CUDA_VISIBLE_DEVICES=2 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATA_DGX1/NorHandv3_mini_v3/test/Images" \
  --output "${PATHLOG}/H087/NorHandv3_mini_v3/images" \
  --save_boxes_dir "${PATHLOG}/H087/NorHandv3_mini_v3/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.7 0.5 \
  --map True \
  --img_ext jpg \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/42_2026-01-17_DGX_JOB_01/crnn_best.torch" \
  --htr_charset "${root}/DATA_DGX1/NorHandv3_mini_v3/charset_norhand_v3_mini.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage2_results.txt" \
  --text TrainOn_NorHandv3_mini_v3-TestOn_NorHandv3_mini_v3 \
>> "$output_file"