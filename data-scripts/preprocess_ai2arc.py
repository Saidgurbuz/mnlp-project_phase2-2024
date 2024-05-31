import os
import json
from datasets import load_dataset

def preprocess_ai2_arc(entry):
    subject = "AI2ARC" # There isn't a subject field in the dataset
    question_text = entry['question']
    choices = entry['choices']['text']
    labels = entry['choices']['label']
    correct_answer = entry['answerKey']

    # Pairing labels with their corresponding text
    choice_pairs = zip(labels, choices)
    formatted_choices = "\n".join([f"{label}. {text}" for label, text in choice_pairs])

    # Building the question format
    formatted_question = f"Question: {question_text}\n\nOptions:\n{formatted_choices}\n\nAnswer:"

    return {
        "subject": subject,
        "question": formatted_question,
        "answer": correct_answer
    }

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

def process_datasets(configs):
    all_data = []
    for config_name in configs:
        print(f"Processing {config_name}...")
        # Adjust split as needed - 'train', 'validation', 'test'
        dataset = load_dataset("allenai/ai2_arc", config_name, split='test') 
        preprocessed_data = [preprocess_ai2_arc(entry) for entry in dataset]
        all_data.extend(preprocessed_data)
        print(f"Finished processing {config_name}.")
    return all_data

if __name__ == "__main__":
    configs = ['ARC-Challenge', 'ARC-Easy']
    combined_data = process_datasets(configs)
    output_file_path = "mcq/ai2_arc/mcqa_ai2arc_test.jsonl"
    save_to_jsonl(combined_data, output_file_path)
