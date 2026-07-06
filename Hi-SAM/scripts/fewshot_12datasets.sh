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
file=12DatasetEval_fewshot.py

root="/home/$USER/PROJECTS"
main_script="${root}/codes/Hi-SAM_Doc/${file}"


PATHLOG="${root}/logs/Hi-SAM_Doc/sample_output2"
echo "path log :"
echo ${PATHLOG}

output_file="${PATHLOG}/out.txt"

export PYTHONPATH=/proj/document_analysis/users/x_gapat/codes/Hi-SAM_Doc/

############################# Train on READ #############################
################# H060
python "$main_script" \
>> "$output_file"