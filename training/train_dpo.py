import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import HfArgumentParser, TrainingArguments, set_seed

from utils_dpo import create_and_prepare_model, create_datasets
from models.model_dpo import AutoDPOModelForCausalLM


# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="gpt2",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    max_seq_length: Optional[int] = field(default=1024, metadata={"help": "The context length of the model."})
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={
            "help": "chatml|zephyr|none. Pass `none` if the dataset is already formatted with the chat template."
        },
    )
    use_flash_attn: Optional[bool] = field(default=False)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="c_attn,c_proj",
        # The list below is for the model in the SFT example
        # default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )

@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        default="data/dpo_preference_example.jsonl",
        metadata={"help": "The preference dataset to use."},
    )
    val_perc: float = field(default=0.1)

@dataclass
class TrainingArguments:
    device: str = field(default="cpu")
    lr: float = field(default=1e-4)
    batch_size: int = field(default=4)
    seed: int = field(default=239)
    num_train_epochs: int = field(default=100)

class DPOTrainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, training_args):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.num_epochs = training_args.num_train_epochs

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=training_args.lr)

    def train(self):
        for e in range(self.num_epochs):
            for batch in self.train_dataset:
                chosen_ref_logprobs, rejected_ref_logprobs = self.model.get_ref_logprobs(batch, self.tokenizer)
                chosen_logprobs, rejected_logprobs = self.model.get_logprobs(batch, self.tokenizer)
                rewards = self.model.prediction_step_reward(chosen_logprobs, rejected_logprobs, chosen_ref_logprobs,
                                                            rejected_ref_logprobs)
                loss = -torch.sum(F.logsigmoid(rewards["chosen_rewards"] - rewards["rejected_rewards"]))
                print(f"Loss: {loss}")

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # To test that loss goes down, uncomment
                # break

    def save_model(self):
        pass


def main(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # model
    pretrained_model, peft_config, tokenizer = create_and_prepare_model(model_args)
    pretrained_model = pretrained_model.to(torch.device(training_args.device))

    dpo_model = AutoDPOModelForCausalLM(pretrained_model, peft_config, model_args.max_seq_length)
    dpo_model = dpo_model.to(torch.device(training_args.device))

    # gradient ckpt
    # dpo_model.config.use_cache = not training_args.gradient_checkpointing
    # training_args.gradient_checkpointing = training_args.gradient_checkpointing
    # if training_args.gradient_checkpointing:
    #     training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}

    # datasets
    # TODO: load the correct ones
    train_dataset, eval_dataset = create_datasets(data_args.dataset_name, training_args.batch_size,
                                                  data_args.val_perc, apply_chat_template=False)

    # trainer
    trainer = DPOTrainer(dpo_model, tokenizer, train_dataset, eval_dataset, training_args)

    # train
    # checkpoint = None
    # if training_args.resume_from_checkpoint is not None:
    #     checkpoint = training_args.resume_from_checkpoint
    # trainer.train(resume_from_checkpoint=checkpoint)
    trainer.train()

    # saving final model
    trainer.save_model()


if __name__ == "__main__":
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     # If we pass only one argument to the script and it's the path to a json file,
    #     # let's parse it to get our arguments.
    #     model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    # else:
    #     model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_args = ModelArguments()
    data_args = DataArguments()
    training_args = TrainingArguments()
    main(model_args, data_args, training_args)
