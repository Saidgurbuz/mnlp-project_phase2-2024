import os
import json
import random
import numpy as np
from datasets import load_dataset

subjects = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 
            'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
            'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 
            'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 
            'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 
            'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 
            'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 
            'high_school_physics', 'high_school_psychology', 'high_school_statistics', 
            'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 
            'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 
            'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 
            'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 
            'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 
            'virology', 'world_religions']

# subjects = ['machine_learning'] - to test the script

def preprocess_mmlu(entry):
    subject = entry['subject']
    question = entry['question']
    choices = entry['choices']
    correct_answer = entry['answer']

    # Building the question format same to the one given on Github
    formatted_question = "Question: " + question + "\n\nOptions:"
    for idx, choice in enumerate(choices):
        formatted_question += f"\n{chr(65+idx)}. {choice}"

    # Convert the int to the corresponding option
    answer = chr(65 + correct_answer)  

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
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    # Collect all test data for MMLU from the test split
    all_data = []
    
    for subject in subjects:
        # In this script we processes the test data only. Adjust split as needed.
        dataset = load_dataset("cais/mmlu", subject, split='test')

        # Preprocess and collect all formatted entries
        subject_data = [preprocess_mmlu(entry) for entry in dataset]
        all_data.extend(subject_data)


    output_file_path = "mcq/mmlu/mmlu.jsonl" 

    save_to_jsonl(all_data, output_file_path)

