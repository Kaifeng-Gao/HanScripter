#!/bin/bash

#SBATCH --job-name=translate
#SBATCH --time=7-00:00:00
#SBATCH --partition=week
#SBATCH --mail-type=ALL

module purge
module load miniconda
conda activate Llama

python translate.py
