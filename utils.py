import torch
import transformers, evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel


def load_peft_model_and_tokenizer(model_path, new_model_path):
    """Loads a PEFT model from the given path and merges it with the base model."""
    base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    model = PeftModel.from_pretrained(base_model, new_model_path)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer

def load_base_model_and_tokenizer(model_path):
    """Loads the base transformer model and tokenizer from the given path."""
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    return model, tokenizer


def load_model(model_path, new_model_path=None, finetune=False):
    if finetune:
        return load_peft_model_and_tokenizer(model_path, new_model_path)
    else:
        return load_base_model_and_tokenizer(model_path)