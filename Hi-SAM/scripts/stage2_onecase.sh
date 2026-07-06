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

root=/home/gayapath/PROJECTS
main_script="${root}/codes/Hi-SAM_Doc/${file}"


PATHLOG="${root}/logs/Hi-SAM_Doc/sample_output"
echo "path log :"
echo ${PATHLOG}

output_file="${PATHLOG}/out.txt"

export PYTHONPATH=/proj/document_analysis/users/x_gapat/codes/Hi-SAM_Doc/

############################# Train on READ #############################
################# H060
PRETRAINE_MODEL=60_2025-10-01_ID_14189314
CUDA_VISIBLE_DEVICES=3 python "$main_script" \
  --checkpoint "${root}/logs/Hi-SAM_Doc/${PRETRAINE_MODEL}/saved_model/READ_2016_best_mAP.pth" \
  --model-type vit_h \
  --pretrained_path "${root}/logs/Hi-SAM_Doc/pretrained_checkpoint" \
  --input "${root}/DATASETS/Nuremberg_Letterbooks/Band2/test/Images" \
  --output "${root}/logs/Hi-SAM_Doc/sample_output/H060/Nuremberg_Letterbooks/Band2/images" \
  --save_boxes_dir "${root}/logs/Hi-SAM_Doc/sample_output/H060/Nuremberg_Letterbooks/Band2/boxes" \
  --dataset ctw1500 \
  --exp_id 0 \
  --nms 0.6 0.6 \
  --map True \
  --img_ext jpg \
  --htr_crnn_model_path "${root}/logs/CRNN_Center_loss/ATS_HTR_Training/4_2025-12-02_ID_14885224/crnn_best.torch" \
  --htr_charset "${root}/DATASETS/READ_2016/charset.txt" \
  --htr_bb_sorting multi-col-sorting \
  --results_log "${PATHLOG}/stage2_results_tmp.txt" \
  --text TrainOn_READ2016-TestOn_Nuremberg_Band2 \
>> "$output_file"
Rst Nbg-Briefbücher-Nr 2_0126_left
