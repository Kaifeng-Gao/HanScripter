#!/bin/bash

#SBATCH --job-name=evaluate
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1

module purge
module load miniconda
conda activate cpsc577

python inference.py