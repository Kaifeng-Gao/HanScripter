#!/bin/bash

#SBATCH --job-name=evaluate
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1

module purge
module load miniconda
conda activate cpsc577

python llama_evaluate.py --new_model_path "./results/Llama-3-Han" --finetune True
python llama_evaluate.py --new_model_path "./results/Llama-3-Han-Instruct" --finetune True
python llama_evaluate.py
python gemini_evaluate.py