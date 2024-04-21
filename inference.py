import yaml
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from utils import load_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers, evaluate


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Load configurations from YAML file
config = load_config('config.yaml')

# Setup configurations
model_cfg = config['model_config']

# Initialize configuration with specified path
model_path = model_cfg['model_path']
new_model_path =  model_cfg['new_model_path']
dataset_path = model_cfg['dataset_path']
dataset_config = model_cfg['dataset_config']

model, tokenizer = load_model(model_path, new_model_path, finetune=True)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16}
)
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Initialize test dataset
dataset = load_dataset(dataset_path, dataset_config)
dataset_train, dataset_test = dataset.select_columns(["classical", "english"]).values()
dataset_examples = dataset_test.select(range(5))
dataset_predict = dataset_test.select(range(5, len(dataset_test)))

prompt_template = "Classical Chiense: {classical} \nEnglish: {english}"
prompt_examples = "\n\n".join([prompt_template.format(**row) for row in dataset_examples])

# Initialize test prompts and reference translations
prompts = []
references = []
for row in dataset_predict:
    message = []
    prompt= prompt_examples + "\n\n"\
            + "Based on the above examples, translate the classical chinese sentence below into english:" + "\n"\
            + prompt_template.format(classical=row["classical"], english="")[:-1]
    message.append({"role": "user", "content": prompt})
    message = pipeline.tokenizer.apply_chat_template(
        message, 
        tokenize=False, 
        add_generation_prompt=True)
    reference=row["english"]
    prompts.append(message)
    references.append(reference)

# Evaluation
outputs = pipeline(
    prompts,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

predictions = [output[0]['generated_text'][len(prompt):] for output, prompt in zip(outputs, prompts)]
print("------------- example -------------")
print(prompts[0])
print(predictions[0])
print(references[0])

sacrebleu = evaluate.load("sacrebleu")
sacrebleu_results=sacrebleu.compute(predictions=predictions, references=references)
print("------------- sacrebleu_results -------------")
print(sacrebleu_results)