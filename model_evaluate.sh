#!/bin/bash

#SBATCH --job-name=evaluate
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1

module purge
module load miniconda
conda activate cpsc577

python llama_evaluate.py --new_model_path "./results/Llama-3-Han-0422" --finetune True --num_shots 5
python llama_evaluate.py --new_model_path "./results/Llama-3-Han-0423" --finetune True --num_shots 5
python llama_evaluate.py --num_shots 5

python llama_evaluate.py --new_model_path "./results/Llama-3-Han-0422" --finetune True --num_shots 3
python llama_evaluate.py --new_model_path "./results/Llama-3-Han-0423" --finetune True --num_shots 3
python llama_evaluate.py --num_shots 3

python llama_evaluate.py --new_model_path "./results/Llama-3-Han-0422" --finetune True --num_shots 1
python llama_evaluate.py --new_model_path "./results/Llama-3-Han-0423" --finetune True --num_shots 1
python llama_evaluate.py --num_shots 1

python llama_evaluate.py --new_model_path "./results/Llama-3-Han-0422" --finetune True --num_shots 0
python llama_evaluate.py --new_model_path "./results/Llama-3-Han-0423" --finetune True --num_shots 0
python llama_evaluate.py --num_shots 0

# python llama_evaluate.py --new_model_path "./results/Llama-3-Han" --finetune True
# python llama_evaluate.py --new_model_path "./results/Llama-3-Han-Instruct" --finetune True
# python llama_evaluate.py
# python gemini_evaluate.py