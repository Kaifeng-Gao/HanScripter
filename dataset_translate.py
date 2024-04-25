import os
from datasets import load_dataset
import time
from gemini_loader import GeminiLoader

CONFIG_PATH = 'config.yaml'

gemini_loader = GeminiLoader(CONFIG_PATH)
model = gemini_loader.create_model()

def translate_modern(sample):
    '''
    Use LLM to translate modern chinese to english in batch
    '''
    attempt = 0
    max_attempts = 5  # Set a limit to prevent infinite loops
    success = False

    modern = sample['modern']
    while attempt < max_attempts and not success:
        try:
            prompt = "Translate the following sentences from Chinese to English:\n" + "\n".join(f"{i+1}. {sentence}" for i, sentence in enumerate(modern)) + "\n\nPlease provide the translations in the format:\n\n<number>. <translation>"
            # Assuming model.generate_content(prompt) is a method call to an LLM API or similar service
            response = model.generate_content(prompt)
            translations = [None for i in modern]
            for line in response.text.strip().split('\n'):
                number, translation = line.split('. ', 1)
                translations[int(number)-1] = translation
            sample['english'] = translations
            success = True  # Mark success to exit the loop
        except Exception as e:
            print(f"{type(e).__name__}: {e}")
            attempt += 1
            if attempt < max_attempts:
                print(f"Attempt {attempt} failed. Retrying in 60 seconds...")
                time.sleep(60)  # Wait for 60 seconds before the next attempt
    if not success:
        print("Max attempts reached. Failed to translate due to repeated errors.")

    return sample


dataset = load_dataset("xmj2002/Chinese_modern_classical", split="train")
dataset_size = len(dataset)
subset_cnt = 0
step_size = 100000
for i in range(0, dataset_size, step_size):
    print(f"Translating {i} to {min(i+step_size, dataset_size)}")
    subset_cnt += 1
    dataset = load_dataset("xmj2002/Chinese_modern_classical", split=f"train[{i}:{min(i+step_size, dataset_size)}]")
    dataset = dataset.map(translate_modern, batched=True, batch_size=10)
    dataset.save_to_disk(f"./dataset/subset_{subset_cnt}")
