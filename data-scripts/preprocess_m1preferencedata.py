import os
import json
import random
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file_path):
    print("Loading the original dataset...")
    with open(file_path, 'r') as file:
        return json.load(file)

def save_to_jsonl(data, filename):
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    print(f"Saving data to {filename}...")
    with open(filename, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')
    print(f"Data saved successfully to {filename}.")

def reformat_data(data):
    print("Transforming the data...")
    reformatted_data = []
    
    # Loop through each question in the dataset
    for entry in data:
        # Remove "Question: " from the beginning
        prompt = entry["question_complete"].replace("Question: ", "", 1) 
        
        # Loop through each preference pair for the question
        for pref in entry["preference"]:
            # Determine chosen and rejected based on the 'overall' preference

            # Here we can add other conditions to better handle the pairs, maybe by aggregating
            # the criteria in order to also compare correctness, clarity etc. 
            if pref["overall"] == "A":
                chosen = pref["A"]
                rejected = pref["B"]
            elif pref["overall"] == "B":
                chosen = pref["B"]
                rejected = pref["A"]
            else:
                # If neither A nor B is chosen as overall, we skip this entry
                continue  
            
            # I found a lot of examples where the chosen and rejected answer are the same 
            # which is not a good preference pair, so we skip this entry
            if chosen == rejected:
                continue
            
            reformatted_entry = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            }
            reformatted_data.append(reformatted_entry)
    
    return reformatted_data

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    # Path to the input JSON file
    input_file_path = "M1_preference_data_15052024.json"

    # Path to the output JSONL files
    output_file_path = "dpo/m1preferencedata/m1.jsonl" 
    train_output_file_path = "dpo/m1preferencedata/train_m1.jsonl"
    validation_output_file_path = "dpo/m1preferencedata/validation_m1.jsonl"

    original_data = load_data(input_file_path)
    transformed_data = reformat_data(original_data)

    # Split the data into training and validation sets
    train_data, validation_data = train_test_split(transformed_data, test_size=0.3, random_state=42)

    save_to_jsonl(train_data, train_output_file_path)
    save_to_jsonl(validation_data, validation_output_file_path)
    save_to_jsonl(transformed_data, output_file_path)

    print("Data transformation and splitting complete.")
