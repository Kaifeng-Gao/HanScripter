from datasets import load_dataset
from gemini_loader import GeminiLoader
import os
import yaml
import transformers, evaluate
import argparse
import sys
import time

CONFIG_PATH = 'config.yaml'

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def parse_args():
    parser = argparse.ArgumentParser(description="Load and fine-tune a model")
    parser.add_argument('--model_path', type=str, help='Path to the base model')
    parser.add_argument('--new_model_path', type=str, help='Path for saving the fine-tuned model')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('--dataset_config', type=str, help='Dataset configuration')
    parser.add_argument('--num_shots', type=int, help='Number of shots in instruction')
    parser.add_argument('--finetune', type=bool, help='whether to use finetuned model')
    return parser.parse_args()

# Load configurations from YAML file

config = load_config(CONFIG_PATH)

# Setup configurations
args = parse_args()
model_cfg = config['model_config']
eval_cfg = config['eval_config']
access_token = config['access_token']

# Initialize Configurations
dataset_path = args.dataset_path if args.dataset_path else model_cfg['dataset_path']
dataset_config = args.dataset_config if args.dataset_config else model_cfg['dataset_config']
num_shots = args.num_shots if args.num_shots else eval_cfg['num_shots']

print('=' * 50)
print("------------- eval_config -------------")
print(f"Configuration used:")
print(f"Model Path: Gemini-Pro")
print(f"Dataset Path: {dataset_path}")
print(f"Dataset Config: {dataset_config}")
print(f"Number of Shots: {num_shots}")

# Load Gemini
gemini_loader = GeminiLoader(CONFIG_PATH)
model = gemini_loader.create_model()

# Load Dataset
dataset = load_dataset(dataset_path, dataset_config)
dataset_train, dataset_test = dataset.select_columns(["classical", "english"]).values()
dataset_examples = dataset_test.select(range(num_shots))
dataset_predict = dataset_test.select(range(num_shots, len(dataset_test)))

prompt_template = "Classical: {classical}\nEnglish: {english}"
if num_shots == 0:
    prompt_examples = ""
else:
    prompt_examples = "\n\n".join([prompt_template.format(**row) for row in dataset_examples])

# Initialize test prompts and reference translations
prompts = []
references = []
for row in dataset_predict:
    prompt= prompt_examples + "\n"\
            + "Translate the following sentence from Classical Chinese into English. Provide only the translation:" \
            + prompt_template.format(classical=row["classical"], english="")[:-1]
    reference=row["english"]
    prompts.append(prompt)
    references.append([reference])

# prompts, references = prompts[:10], references[:10]
print("# of test samples:", len(prompts))
predictions = []
for prompt in prompts:
    attempt = 0
    max_attempts = 5  # Set a limit to prevent infinite loops
    success = False
    while attempt < max_attempts and not success:
        try:
            response = model.generate_content(prompt)
            predictions.append(response.text)
            success = True
        except Exception as e:
            print(f"{type(e).__name__}: {e}")
            attempt += 1
            if attempt < max_attempts:
                print(f"Attempt {attempt} failed. Retrying in 60 seconds...")
                time.sleep(60)  # Wait for 60 seconds before the next attempt

# Evaluation
sacrebleu = evaluate.load("sacrebleu")
sacrebleu_results=sacrebleu.compute(predictions=predictions, references=references)
print("------------- sacrebleu_results -------------")
print(sacrebleu_results)

meteor = evaluate.load('meteor')
meteor_results = meteor.compute(predictions=predictions, references=references)
print("------------- meteor_results -------------")
print(meteor_results)

chrf = evaluate.load("chrf")
chrf_results = chrf.compute(predictions=predictions, references=references)
print("------------- chrf_results -------------")
print(chrf_results)

# Example
print("------------- example -------------")
for i in range(10):
    print("system:", predictions[i])
    print("reference:", references[i][0])
