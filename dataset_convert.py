from datasets import load_dataset
import os
from gemini_loader import GeminiLoader
import random
import time


CONFIG_PATH = 'config.yaml'
DATASET_PATH = "KaifengGGG/WenYanWen_English_Parrallel"
DATASET_SIZE = 10000
STEP_SIZE = 1000
TEMPLATES = [
    # without LLM augmented
    "[INST] Translate the following Classical Chinese text into modern Chinese: {classical} [/INST] {modern}",
    "[INST] Translate the following Classical Chinese text into English: {classical} [/INST] {english}",
    "[INST] Explain the meaning of this Classical Chinese passage in English: {classical} [/INST] {english}",
    "[INST] Convert the following English text into Classical Chinese: {english} [/INST] {classical}",
    "[INST] First, translate the Classical Chinese text into modern Chinese and then into English. Classical Chinese: {classical} [/INST] {modern} Then, {english}", #COT
    "[INST] Give the modern Chinese and English translation for the following Classical Chinese passage: {classical} [/INST] Modern Chinese: {modern} English: {english}",
    "[INST] Identify the key themes in this Classical Chinese text and explain them in English: {classical} [/INST] {english}",
    "[INST] For each of the following sets of Classical Chinese and Modern Chinese sentences, provide an English translation that captures the essence of both: Classical: {classical} Modern: {modern} [/INST] {english}",
    "[INST] Convert the following modern Chinese text into Classical Chinese: {modern} [/INST] {classical}",
    "[INST] Translate the following passage from Classical Chinese to both modern Chinese and English, focusing on preserving the emotional tone in both translations: {classical} [/INST] Modern Chinese: {modern} English: {english}",
    "[INST] Provide a modern Chinese interpretation of the following Classical Chinese passage: {classical} [/INST] {modern}",
    "[INST] Render the following Classical Chinese text into fluent English: {classical} [/INST] {english}",
    "[INST] Rewrite the following English passage in the style of Classical Chinese: {english} [/INST] {classical}",
    "[INST] Translate the given Classical Chinese text into modern Chinese, then provide an English translation that captures the nuances of both versions. Classical Chinese: {classical} [/INST] Modern Chinese: {modern} English Translation: {english}",
    "[INST] Analyze the following Classical Chinese passage and provide both a modern Chinese and an English translation: {classical} [/INST] Modern Chinese Interpretation: {modern} English Translation: {english}",
    "[INST] Transpose the following Modern Chinese text into Classical Chinese: {modern} [/INST] {classical}",
    "[INST] Translate to English, please: {classical} [/INST] {english}", 
    "[INST] Kindly elucidate the central themes in this passage in English: {classical} [/INST] {english}",
    "[INST] I need this in both modern chinese phrasing and English - could you help? {classical} [/INST] Contemporary: {modern} English version: {english}",
    "[INST] Take each pair of old and new sentences and unite them into a single translation that captures the core meaning: Archaic: {classical} Updated: {modern} [/INST] {english}",
    "[INST] 翻译: {classical} [/INST] {english}",
    "[INST] 将以下文言文翻译成现代汉语: {classical} [/INST] {modern}",
    "[INST] 请将下面的文言文翻译成英语: {classical} [/INST] {english}",
    "[INST] 用英语解释这段文言文的意思: {classical} [/INST] {english}",
    "[INST] 将以下英文转换为文言文: {english} [/INST] {classical}",
    "[INST] 首先, 将文言文翻译成现代汉语, 然后再翻译成英语。文言文: {classical} [/INST] 现代汉语: {modern} 英语: {english}",
    "[INST] 请给出以下文言文的现代汉语和英语翻译: {classical} [/INST] 现代汉语: {modern} 英语: {english}",
    "[INST] 对于下列每组文言文和现代汉语句子, 请提供一个能够准确表达两者本质的英语翻译: 文言文: {classical} 现代汉语: {modern} [/INST] {english}",
    "[INST] 将以下现代汉语文本转换为文言文: {modern} [/INST] {classical}",
    "[INST] 请提供这段文言文的现代汉语解释: {classical} [/INST] {modern}",
    "[INST] 将以下文言文流畅地翻译成英语: {classical} [/INST] {english}",
    "[INST] 分析以下文言文, 并提供现代汉语和英语翻译: {classical} [/INST] 现代汉语解释: {modern} 英语翻译: {english}",
    "[INST] 对于下面每对文言文和现代汉语段落, 请创作一个能够体现核心信息的英语翻译: 文言文: {classical} 现代汉语: {modern} [/INST] {english}",
    "[INST] 将以下现代汉语文本改写为文言文: {modern} [/INST] {classical}"
]

TEMPLATES_AUGMENT = [
    # with LLM augmented
    "[INST] Analyze the grammatical structure of the following Classical Chinese sentence and provide its English translation: {classical} [/INST] Grammatical analysis: [...], English translation: {english}",
    "[INST] Given the Classical Chinese text: {classical}, provide a step-by-step translation process to arrive at the English translation. [/INST] Step 1: [...], Step 2: [...], [...], Final English translation: {english}",
    "[INST] Identify the tone and emotion conveyed in the Classical Chinese text and explain how it is achieved in English: {classical} [/INST] Tone and emotion: [...], Explanation in English: {english}",
    "[INST] Highlight and translate the key terms in the Classical Chinese text: {classical}, and explain their significance in the context of the passage. [/INST] Key terms and their translations: [...], Final English translation: {english}",
    "[INST] From the Classical Chinese script: {classical}, document the translation approach to render an English version. [/INST] Initial step: [...], Following steps: [...], English rendition: {english}",
    "[INST] Break down the structure and provide an English interpretation of the presented Classical Chinese phrase: {classical} [/INST] English interpretation: {english}, Structural analysis: [...]",
    "[INST] For the Classical Chinese excerpt: {classical}, detail the steps needed to accurately translate it into English. [/INST] Completed English translation: {english}, Step 1: [...], Step 2: [...], [...]"
]

gemini_loader = GeminiLoader(CONFIG_PATH)
model = gemini_loader.create_model()

def format_sample_with_template(sample, augment):
    if augment:
        selected_template = random.choice(TEMPLATES + TEMPLATES_AUGMENT)
    else:
        selected_template = random.choice(TEMPLATES)
    required_keys = set()
    placeholders = ["classical", "modern", "english"]
    for placeholder in placeholders:
        if '{' + placeholder + '}' in selected_template:
            required_keys.add(placeholder)
    relevant_data = {k: sample.get(k, '') for k in required_keys}
    # Using exception handling to ignore placeholders if data is missing in the sample
    try:
        formatted_text = selected_template.format_map(relevant_data)
    except KeyError as e:
        print(f"Error processing template: {e}")
        formatted_text = selected_template
        
    # Augmented the instruction - answer pair by filling the blanks using LLM
    if "..." in selected_template:
        attempt = 0
        max_attempts = 5  # Set a limit to prevent infinite loops
        success = False
        while attempt < max_attempts and not success:
            try:
                # prompt = "I have some corrupted text. Text: '" + formatted_text + "'.Fill in the '[...]' in the text with your the best of your knowledge and the context. Only Return the text in original format with all parts filled in"
                prompt = f""" For the given sample, use your knowledge and the context provided to complete the task/response according to the instructions within the brackets "[INST] ... [/INST]". The task may involve analyzing grammatical structures, translating texts into Modern Chinese and English, explaining translation processes, identifying tones and emotions, or discussing grammatical and vocabulary differences between languages. Ensure you adhere to the format outlined in the instruction and address all components comprehensively.
                Sample to be completed: "{formatted_text}"
                Identify the task type and fill in the missing information "[...]" with the appropriate content. Ensure your response is complete, following the original instruction format. Return the original instructions and the completed response in format like "[INST] instruction [/INST] response".
                """
                response = model.generate_content(prompt)
                formatted_text = response.text
                success = True
            except Exception as e:
                print(f"{type(e).__name__}: {e}")
                attempt += 1
                if attempt < max_attempts:
                    print(f"Attempt {attempt} failed. Retrying in 60 seconds...")
                    time.sleep(60)  # Wait for 60 seconds before the next attempt
    return formatted_text


def convert_to_conversation(text):
    snippets = text.split('<s>')  # Split and remove the first and last useless parts.
    messages = []
    
    for snippet in snippets:
        if '[INST]' in snippet and '[/INST]' in snippet:
            user, model = snippet.split('[/INST]', 1)
            user = user.replace('[INST]', '').strip()
            model = model.replace('</s>', '').strip()
            if user:
                messages.append({"role": "user", "content": user})
            if model:
                messages.append({"role": "assistant", "content": model})
    
    return messages



def template_dataset(sample, augment):
    formatted_text = format_sample_with_template(sample, augment)
    sample["text"] = "<s>" + formatted_text + "</s>"
    sample["messages"] = convert_to_conversation(sample["text"])
    return sample


subset_cnt = 0

for i in range(0, DATASET_SIZE, STEP_SIZE):
    subset_cnt += 1
    dataset = load_dataset(DATASET_PATH, split=f"train[{i}:{min(i+STEP_SIZE, DATASET_SIZE)}]")
    dataset_mistral = dataset.map(template_dataset, fn_kwargs={'augment': False})
    dataset_mistral.save_to_disk(f"./dataset_instruct_message/subset_{subset_cnt}")

