import os
import json
import random
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_and_process_datasets(community_list):
    datasets = {}

    for community in community_list:
        print(f"Loading dataset...")
        datasets[community] = load_dataset(
            "flax-sentence-embeddings/stackexchange_titlebody_best_and_down_voted_answer_jsonl",
            community
        )

    all_entries = []
    for name, data in datasets.items():
        print(f"Processing {name} data...")
        for entry in data['train']:
            formatted_entry = {
                "prompt": entry["title_body"],
                "chosen": entry["upvoted_answer"],
                "rejected": entry["downvoted_answer"]
            }
            all_entries.append(formatted_entry)

    train_entries, validation_entries = train_test_split(
        all_entries, test_size=0.3, random_state=42) 

    train_dir = 'dpo/stackexchange/train_stackexchange.jsonl'
    validation_dir = 'dpo/stackexchange/validation_stackexchange.jsonl'
    os.makedirs(os.path.dirname(train_dir), exist_ok=True)
    with open(train_dir, 'w') as outfile:
        for entry in train_entries:
            json.dump(entry, outfile)
            outfile.write('\n')

    os.makedirs(os.path.dirname(validation_dir), exist_ok=True)
    with open(validation_dir, 'w') as outfile:
        for entry in validation_entries:
            json.dump(entry, outfile)
            outfile.write('\n')

    print("Data split complete.")


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    stem_communities = [
        'physics', 'bioinformatics', 'electronics', 'mathoverflow', 'codereview',
        'cs', 'cstheory', 'datascience', 'matheducators',
        'engineering', 'ai', 'cseducators', 'iot', 'softwareengineering', 'stats', 'networkengineering',
        'scicomp', 'robotics', 'devops', 'astronomy', 'askubuntu', 'apple',
        'serverfault', 'security', 'webapps', 'webmasters'
    ]

    load_and_process_datasets(stem_communities)
