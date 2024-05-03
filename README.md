# HanScripter
Unveiling the Wisdom of Classical Chinese with Llama

## Dataset

The [WenYanWen_English_Parallel](https://huggingface.co/datasets/KaifengGGG/WenYanWen_English_Parrallel) dataset is a multilingual parallel corpus in Classical Chinese (Wenyanwen), modern Chinese, and English. This dataset is created by Kaifeng Gao, Ke Lyu and Abbey Yuan.

The code used to create and process the dataset can be found [here](https://github.com/Kaifeng-Gao/WenYanWen_English_Parallel).

The final dataset contains four subsets:
- **default**: A parallel translation dataset containing 972,467 translation pairs.
- **instruct**: An instruction-tuning dataset consisting of prompt/answer pairs created from a 10,000-sample of the default subset.
- **instruct-augment**: Similar to the instruct subset, with the distinction that the prompt/answer pairs have been augmented by Gemini-Pro.
- **instruct-large**: An instruction-tuning dataset that includes all samples from the default dataset.
\end{itemize}

## Dependency

### One-step Installation

```bash
conda env create -f environment.yml
```

### Manual Installation

1. **Create and Activate Environment**
   ```bash
   conda create --name llama python=3.10
   conda activate llama
   ```

2. **Install PyTorch with CUDA Support**
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

3. **Test CUDA (Optional)**

   Run the CUDA test Python script
   ```python
   import torch
   torch.version.cuda
   ```
   If it returns a cuda version, the PyTorch with CUDA support is installed successfully.

4. **Install Hugging Face Libraries**

   - Transformers: `conda install conda-forge::transformers`
   - Datasets: `conda install -c huggingface -c conda-forge datasets`
   - Accelerate: `conda install -c conda-forge accelerate`
   - bitsandbytes (for efficient CUDA operations): `pip install bitsandbytes`

5. **Install Jupyter Notebook (Optional)**

   ```bash
   conda install jupyter
   ```

6. **Install Additional Dependencies with pip**
   - For QLoRA finetuning: `pip install trl peft`
   - For Evaluation: `pip install evaluate bert_score nltk sacrebleu`
   - For Gemini: `pip install -q -U google-generativeai`
   - For OPENAI GPT: `pip install openai`

## Quick Start

Before starting, first copy the `config.yaml.template` file and rename it to `config.yaml`. Then, modify the `config.yaml` file according to your needs.

Hugging Face token with access to Llama3 model ([apply here](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)) is required for fine-tuning. Google and OPENAI tokens are needed for evaluation on Gemini-pro and GPT-4-Turbo.

### SFT with QLoRA

```bash
python finetune.py
# or sbatch finetune.sh if on Yale High Performance Computing
```

### Evaluation
```bash
python llama_evaluate.py --new_model_path "./results/Llama-3-Han-0422" --finetune True --num_shots 5
# or sbatch model_evaluate.sh if on Yale High Performance Computing
```



