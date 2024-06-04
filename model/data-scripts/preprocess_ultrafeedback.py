import os
import json
import random
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
 
def transform_entry(entry):
    return {
        'prompt': entry['prompt'],
        'chosen': entry['chosen'][-1]["content"],
        'rejected': entry['rejected'][-1]["content"],
    }
 
def save_to_jsonl(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print(f"Saving data to {filename}...")
    with open(filename, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')
    print(f"Data saved successfully to {filename}.")
 
if __name__ == "__main__":
    seed = 233
    np.random.seed(seed)
    random.seed(seed)
 
    print("Loading dataset...")
    dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")
    print("Dataset loaded successfully.")
 
    print("Transforming data...")
    transformed_data = [transform_entry(entry) for entry in dataset['train']]
    print("Data transformation complete.")
 
    print("Splitting data into train and validation sets...")
    train_data, val_data = train_test_split(transformed_data, test_size=0.1, random_state=233)
    print("Data split complete.")
 
    save_to_jsonl(train_data, 'dpo/ultrafeedback/train_ultrafeedback.jsonl')
    save_to_jsonl(val_data, 'dpo/ultrafeedback/validation_ultrafeedback.jsonl')

    # save_to_jsonl(transformed_data, 'dpo/ultrafeedback/ultrafeedback.jsonl')
    print("Data saved!")