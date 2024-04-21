#!/bin/bash

#SBATCH --job-name=translate
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --mail-type=ALL

module purge
module load miniconda
conda activate cpsc577

python finetune.py