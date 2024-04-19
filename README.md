# MistralHanWisdom
Unveiling the Wisdom of Classical Chinese with Mistral

## Dataset

### WenYanWen_English_Parallel
The [WenYanWen_English_Parallel](https://huggingface.co/datasets/KaifengGGG/WenYanWen_English_Parrallel) dataset is a multilingual parallel corpus in Classical Chinese (Wenyanwen), modern Chinese, and English. This dataset is created by Kaifeng Gao, Ke Lyu and Abbey Yuan.

The Classical Chinese and modern Chinese parts are sourced from the [Classical-Modern dataset](NiuTrans/Classical-Modern), while the corresponding English translations are translated from modern Chinese using Gemini Pro.

### WenYanWen_English_Parrallel_Instruct

For supervised finetuning, we also created an instruction version of the dataset: [WenYanWen_English_Parrallel_Instruct](https://huggingface.co/datasets/KaifengGGG/WenYanWen_English_Parrallel_Instruct). In order to leverage instruction fine-tuning, the prompt is surrounded by [INST] and [/INST] tokens. An example is

```python
"""
<s>[INST] Analyze the following Classical Chinese passage and provide both a modern Chinese and an English translation: 历官西南，谙晓先朝遗事，撰《炎徼纪闻》。 [/INST] Modern Chinese Interpretation: 在西南为官，熟知先朝遗事，因撰《炎徼纪闻》。 English Translation: As an official in southwest China, he became familiar with historical events of the previous dynasty. Therefore, he wrote the book Records of Flagrant Rumors from the South.</s>
"""
```
