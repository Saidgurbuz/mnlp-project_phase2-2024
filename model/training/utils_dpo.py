import os
import sys
import string
# Change path so you can run from the root of the project
script_path = os.path.abspath(__file__)
project_root = os.path.join(script_path, os.pardir, os.pardir)
project_root = os.path.normpath(project_root)
sys.path.insert(0, project_root)

import numpy as np
from enum import Enum
from datasets import Dataset, concatenate_datasets
from transformers import PreTrainedTokenizer

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import LoraConfig
import json

DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


class ZephyrSpecialTokens(str, Enum):
    user = "<|user|>"
    assistant = "<|assistant|>"
    system = "<|system|>"
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class ChatmlSpecialTokens(str, Enum):
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]

def read_jsonl_files(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
    return data


def tokenize_and_validate_prompts(tokenizer: PreTrainedTokenizer, prompt, chosen_prompt, rejected_prompt):
    """ 
        Function to check whether input adheres with what DPOTokenizer expects.
    """
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}
    chosen_tokens = build_tokenized_answer(tokenizer, prompt, chosen_prompt)
    rejected_tokens =  build_tokenized_answer(tokenizer, prompt, rejected_prompt)
    prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

    chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
    rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
    prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

    for k, v in prompt_tokens.items():
        prompt_tokens[k] = v[:prompt_len_input_ids]

    num_diff_tokens = sum(
                [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
            )
    num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)

    # Allow only up to 1 token difference and length difference
    return num_diff_tokens <= 1 and num_diff_len <= 1

def transform_dataset(data, tokenizer):
    prompts = []
    chosen_responses = []
    rejected_responses = []
    
    for entry in data:
        if entry["chosen"] == entry["rejected"]:
            continue
        if tokenize_and_validate_prompts(tokenizer, entry["prompt"], entry["chosen"], entry["rejected"]):
            prompts.append(entry["prompt"])
            chosen_responses.append(entry["chosen"])
            rejected_responses.append(entry["rejected"])
    
    return {
        "prompt": prompts,
        "chosen": chosen_responses,
        "rejected": rejected_responses
    }

def create_datasets(tokenizer, train_file_paths, example_file_path, eval_file_paths):
    # Read the data
    combined_data = read_jsonl_files(train_file_paths)
    
    # Transform the data and do sanity checks for dpotrainer
    transformed_data = transform_dataset(combined_data, tokenizer)

    # Create Dataset train/eval Dataset class
    train_dataset = Dataset.from_dict(transformed_data)
    eval_dataset = Dataset.from_dict(transform_dataset(read_jsonl_files(eval_file_paths), tokenizer))
    example_dataset = Dataset.from_dict(transform_dataset(read_jsonl_files([example_file_path]), tokenizer))

    # Get full datasets
    full_train_dataset = train_dataset
    full_test_dataset = concatenate_datasets([eval_dataset, example_dataset])

    return full_train_dataset, full_test_dataset

def create_and_prepare_model(args, data_args):
    if args.use_unsloth:
        from unsloth import FastLanguageModel
    bnb_config = None
    quant_storage_dtype = None

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and args.use_unsloth
    ):
        raise NotImplementedError("Unsloth is not supported in distributed training")

    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
        elif args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    if args.use_unsloth:
        # Load model
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=data_args.max_seq_length,
            dtype=None,
            load_in_4bit=args.use_4bit_quantization,
        )
    else:
        torch_dtype = (
            quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
            torch_dtype=torch_dtype,
        )

    peft_config = None
    chat_template = None
    if args.use_peft_lora and not args.use_unsloth:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
        )

    special_tokens = None
    chat_template = None
    if args.chat_template_format == "chatml":
        special_tokens = ChatmlSpecialTokens
        chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
    elif args.chat_template_format == "zephyr":
        special_tokens = ZephyrSpecialTokens
        chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE

    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            pad_token=special_tokens.pad_token.value,
            bos_token=special_tokens.bos_token.value,
            eos_token=special_tokens.eos_token.value,
            additional_special_tokens=special_tokens.list(),
            trust_remote_code=True,
        )
        tokenizer.chat_template = chat_template
        # make embedding resizing configurable?
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_unsloth:
        # Do model patching and add fast LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
            use_gradient_checkpointing=args.gradient_checkpointing,
            random_state=args.seed,
            max_seq_length=data_args.max_seq_length,
        )

    return model, peft_config, tokenizer



def build_tokenized_answer(tokenizer, prompt, answer):
    """
    # DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
    # Copyright 2023 The HuggingFace Team. All rights reserved.
    Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
    It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
    Reference:
        https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
    """

    full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)
    prompt_input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
    answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

    # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
    full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

    # Prepare input tokens for token by token comparison
    full_input_ids = np.array(full_tokenized["input_ids"])

    if len(full_input_ids) != len(full_concat_input_ids):
        raise ValueError("Prompt input ids and answer input ids should have the same length.")

    # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
    # can be merged together when tokenizing prompt+answer. This could result
    # on the last token from the prompt being different when tokenized on its own
    # vs when done as prompt+answer.
    response_token_ids_start_idx = len(prompt_input_ids)

    # If tokenized prompt is different than both prompt+answer, then it means the
    # last token has changed due to merging.
    if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
        response_token_ids_start_idx -= 1

    prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
    prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

    if len(prompt_input_ids) != len(prompt_attention_mask):
        raise ValueError("Prompt input ids and attention mask should have the same length.")

    answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
    answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

    return dict(
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        input_ids=answer_input_ids,
        attention_mask=answer_attention_mask,
    )