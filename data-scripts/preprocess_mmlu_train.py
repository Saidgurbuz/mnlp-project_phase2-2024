import os
import json
import random
import numpy as np
from datasets import load_dataset

def preprocess_mmlu_auxiliary(entry):
    question_text = entry['question']
    choices = entry['choices']
    correct_answer_index = entry['answer'] 
    subject = entry['subject'] if entry['subject'] else "MMLU-auxiliary-train" 


    formatted_question = "Question: " + question_text + "\n\nOptions:"
    for idx, choice in enumerate(choices):
        formatted_question += f"\n{chr(65 + idx)}. {choice}"

    # Convert the int to the corresponding option label
    answer = chr(65 + correct_answer_index)

    return {
        "subject": subject,
        "question": formatted_question + "\n\nAnswer:",
        "answer": answer
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

if __name__ == "__main__":
    dataset = load_dataset("cais/mmlu", "auxiliary_train", split='train')
    preprocessed_data = [preprocess_mmlu_auxiliary(entry['train']) for entry in dataset]

    output_file_path = "mcq/mmlu/mcqa_mmlu_train.jsonl"
    save_to_jsonl(preprocessed_data, output_file_path)
