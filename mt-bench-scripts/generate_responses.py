import json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import sys
import random

def load_prompts(file_path, sample_size=100):
    """ Load prompts from a JSONL file and randomly sample a specified number of prompts """
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            prompts.append(json.loads(line)['prompt'])
    # Randomly sample the prompts
    if len(prompts) > sample_size:
        prompts = random.sample(prompts, sample_size)
    return prompts


def generate_responses(prompts, model_name):
    """ Generate responses using a specified model """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)

    responses = []
    for prompt in prompts:
        print(f"Generating response for: {prompt}")
        output = generator(prompt, max_length=500, num_return_sequences=1, temperature = 0.0, do_sample = False)
        responses.append(output[0]['generated_text'])
    return responses

def save_responses(prompts, responses, output_file):
    """ Save prompts and responses to a JSONL file """
    with open(output_file, 'w', encoding='utf-8') as file:
        for prompt, response in zip(prompts, responses):
            entry = json.dumps({"prompt": prompt, "response": response})
            file.write(f"{entry}\n")

def main():
    input_file = "../model/datasets/dpo_preference_example.jsonl"
    model_names = ['microsoft/Phi-3-mini-4k-instruct', 'StefanKrsteski/Phi-3-mini-4k-instruct-DPO-EPFL']

    prompts = load_prompts(input_file)
    for model_name in model_names:
        print(f"Processing model: {model_name}")
        responses = generate_responses(prompts, model_name)
        save_responses(prompts, responses, f"{model_name.replace('/', '_')}_responses.jsonl")

if __name__ == '__main__':
    main()
