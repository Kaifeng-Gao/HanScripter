#!/bin/bash

#SBATCH --job-name=evaluate
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1

module purge
module load miniconda
conda activate cpsc577

python llama_evaluate.py \
    --model_path "KaifengGGG/Llama3-8b-Hanscripter" \
    --dataset_path "KaifengGGG/WenYanWen_English_Parallel" \
    --dataset_config "instruct" \
    --num_shots 6

# python gemini_evaluate.py \
#     --dataset_path "KaifengGGG/WenYanWen_English_Parallel" \
#     --dataset_config "instruct" \
#     --num_shots 6

# python gpt_evaluate.py \
#     --dataset_path "KaifengGGG/WenYanWen_English_Parallel" \
#     --dataset_config "instruct" \
#     --num_shots 6