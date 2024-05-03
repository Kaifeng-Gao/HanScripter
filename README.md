# HanScripter
**Classical Chinese (i.e. Wenyanwen 文言文) literature**, including poems, articles, and other literary works, represents a valuable cultural heritage of China and the world. However, due to the vast differences in grammar, semantics, and writing styles between classical and modern Chinese, most people today find it challenging to comprehend and appreciate these ancient texts fully. Although classical Chinese anthologies exist, they are often inaccessible or overwhelming for the general public.

Current large language models, while highly capable in processing modern Chinese, struggle to accurately translate, interpret, and analyze classical Chinese texts. This limitation hinders the preservation and dissemination of these invaluable literary treasures, causing them to become increasingly distant and obscure to modern readers.

To address this issue, we develop a auto-regressive generative language model called **HanScripter**, which can provide efficient and accurate **translation from classical Chinese to English**. Our goal is to make these literary works more accessible and understandable to a wider audience, fostering a deeper appreciation and preservation of this cultural heritage.

## Dataset

The [WenYanWen_English_Parallel](https://huggingface.co/datasets/KaifengGGG/WenYanWen_English_Parrallel) dataset is a multilingual parallel corpus in Classical Chinese (Wenyanwen), modern Chinese, and English. This dataset is created by Kaifeng Gao, Ke Lyu and Abbey Yuan.

The code used to create and process the dataset can be found [here](https://github.com/Kaifeng-Gao/WenYanWen_English_Parallel).

The final dataset contains four subsets:
- **default**: A parallel translation dataset containing 972,467 translation pairs.
- **instruct**: An instruction-tuning dataset consisting of prompt/answer pairs created from a 10,000-sample of the default subset.
- **instruct-augment**: Similar to the instruct subset, with the distinction that the prompt/answer pairs have been augmented by Gemini-Pro.
- **instruct-large**: An instruction-tuning dataset that includes all samples from the default dataset.
\end{itemize}

## Models
### Base Models
We selected the **Meta-Llama-3-8B-Instruct** model as our base model. This model, a member of the Llama 3 family, is an 8 billion-parameter instruction-tuned generative text model that operates in an auto-regressive manner. It utilizes an optimized transformer architecture and has been trained on a diverse dataset compiled from publicly available online sources. Additionally, it has been fine-tuned using techniques such as Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF). The model is accessible through the official Hugging Face repository [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct); however, access may require prior application.

### Fine-tuning
We used QLoRA to perform supervised fine-tuning. QLoRA uses 4-bit quantization to compress a pretrained language model. The LM parameters are then frozen and a relatively small number of trainable parameters are added to the model in the form of Low-Rank Adapters. During finetuning, QLoRA backpropagates gradients through the frozen 4-bit quantized pretrained language model into the Low-Rank Adapters. QLoRA reduces the memory usage of LLM finetuning without performance tradeoffs compared to standard 16-bit model finetuning and allow us to finetune Meta-Llama-3-8B-Instruct model with limit resources. The detailed hyper-parameters used in fine-tuning process can be found in our paper.

### HanScripter Models
We trained three models using different subsets of the [KaifengGGG/WenYanWen_English_Parallel](https://huggingface.co/datasets/KaifengGGG/WenYanWen_English_Parallel) dataset:

- **Hanscripter-subset**: Trained on the *instruct* subset of the dataset.
- **Hanscripter-subset-Gemini**: Trained on the *instruct-augment* subset of the dataset.
- **Hanscripter-full**: Trained on the *instruct-large* subset of the dataset.

All three models can be accessed through Hugging Face [KaifengGGG/Llama3-8b-Hanscripter](https://huggingface.co/KaifengGGG/Llama3-8b-Hanscripter).

## Results
| Model                        | sacreBLEU | chrF   | METEOR | $F_{BERT}$ |
|------------------------------|-----------|--------|--------|------------|
| **Hanscripter-full**             | **15.216** | **0.398** | 37.978 | **0.908**    |
| Hanscripter-subset-Gemini    | 13.346    | **0.398** | 36.858 | 0.905      |
| Hanscripter-subset           | 13.281    | 0.381  | 36.435 | 0.906      |
| Llama3-8b-instruct (base model) | 9.804     | 0.325  | 33.393 | 0.892      |
| Gemini-Pro                   | 13.378    | 0.388  | 37.599 | 0.907      |
| GPT-4-Turbo                  | 13.284    | 0.393  | **38.335** | **0.908**    |

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

Example:
```bash
python llama_evaluate.py \
    --model_path "KaifengGGG/Llama3-8b-Hanscripter" \
    --dataset_path "KaifengGGG/WenYanWen_English_Parallel" \
    --dataset_config "instruct" \
    --num_shots 6
# or sbatch model_evaluate.sh if on Yale High Performance Computing
```



