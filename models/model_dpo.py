import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
from peft import get_peft_model
import gc
from models.model_base import PreTrainedModelWrapper
from models.model_base import create_reference_model

import sys

class AutoDPOModelForCausalLM(PreTrainedModelWrapper):
    """
    An autoregressive model with support for custom modules in addition to the language model.
    This class inherits from `PreTrainedModelWrapper` and wraps a
    `transformers.PreTrainedModel` class. The wrapper class supports classic functions
    such as `from_pretrained`, and `generate`. To call a method of the wrapped
    model, simply manipulate the `pretrained_model` attribute of this class.

    Class attributes:
        - **transformers_parent_class** (`transformers.PreTrainedModel`) -- The parent class of the wrapped model. This
            should be set to `transformers.AutoModelForCausalLM` for this class.
        - **lm_head_namings** (`tuple`) -- A tuple of strings that are used to identify the language model head of the
            wrapped model. This is set to `("lm_head", "embed_out")` for this class but can be changed for other models
            in the future
        - **supported_args** (`tuple`) -- A tuple of strings that are used to identify the arguments that are supported
            by the custom module class you designed. Currently, the supported args are: ______
    """

    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]

    ####################################################################################
    # TODO (Optional): Please put any required arguments for your custom module here
    supported_args = ()

    ####################################################################################

    def __init__(self, pretrained_model, device=torch.device("cuda"), max_len=1024, **kwargs):
        r"""
        Initializes the model.

        Args:
            pretrained_model (`transformers.PreTrainedModel`):
                The model to wrap. It should be a causal language model such as GPT2.
                or any model mapped inside the `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to any `CustomModule` class.
        """
        super().__init__(pretrained_model, **kwargs)

        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        ###########################################################################################
        # TODO (Optional): Please uncomment the following lines to initialize your custom module
        # Make sure CustomModule is repalced with the name of your custom module class
        # Remember that the below lines are just an example
        # You can reanme the class and the variabels to fit your custom module name,
        # just make sure they are consistent in the code
        # =========================================================================================
        # self.is_peft_model = True       
        self.device = device
        self.pretrained_model = self.pretrained_model.to(device)
        self.max_seq_length = max_len
        # check if there are enabled gradients in the model and disable gradient flow 
        for param in self.pretrained_model.parameters():
            if param.requires_grad:
                param.requires_grad = False

        # Not sure why this was necessary, dropped for now
        # custom_kwargs, _, _ = self._split_kwargs(kwargs)
        # self._init_weights()
        ###########################################################################################

    def _init_weights(self, **kwargs):
        """
        Initializes the weights of the custom module. The default initialization strategy is random.
        Users can pass a different initialization strategy by passing the `custom_module_init_strategy`
        argument when calling `.from_pretrained`. Supported strategies are:
            - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `CustomModule` class.
        """
        ###############################################################
        # TODO (Optional): Please implement the initialization strategy for your custom module here
        pass
        ###############################################################

    def state_dict(self, *args, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the custom module
        to the state dictionary of the wrapped model by prepending the key with `custom_module.`.

        IMPORTANT: Make sure to replace `custom_module` with the name of your custom module class name.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            pretrained_model_state_dict = {}

        ###########################################################################################
        # TODO (Optional): Please uncomment the following lines to initialize your custom module
        # Make sure "custom_module" is repalced with the name of your custom module class
        # =========================================================================================
        # peft_state_dict = self.peft_model.state_dict(*args, **kwargs)
        # for k, v in custom_module_state_dict.items():
        #     pretrained_model_state_dict[f"custom_module.{k}"] = v
        ###########################################################################################
        return pretrained_model_state_dict

    def post_init(self, state_dict):
        """
        We add the state dictionary of the custom module to the state dictionary of the wrapped model
        by prepending the key with `custom_module.`. This function removes the `custom_module.` prefix from the
        keys of the custom module state dictionary.
        IMPORTANT: Make sure to replace `custom_module` with the name of your custom module class name.
        """
        if not hasattr(self, 'custom_module'):
            return

        for k in list(state_dict.keys()):
            if "custom_module." in k:
                state_dict[k.replace("custom_module.", "")] = state_dict.pop(k)
        self.custom_module.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for CustomModule models."
                )

            # get the lm_head device
            for name, module in self.pretrained_model.named_modules():
                if any(attribute in name for attribute in self.lm_head_namings):
                    lm_head_device = module.weight.device
                    break

            # put custom_module on the same device as the lm_head to avoid issues
            self.custom_module = self.custom_module.to(lm_head_device)

            def set_device_hook(module, input, outputs):
                r"""
                A hook that sets the device of the output of the model to the device of the first
                parameter of the model.
                Args:
                    module (`nn.Module`):
                        The module to which the hook is attached.
                    input (`tuple`):
                        The input to the module.
                    outputs (`tuple`):
                        The output of the module.
                """
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(lm_head_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)
            self.is_sequential_parallel = True

    def push_to_hub(self, *args, **kwargs):
        """Push the model to the Hugging Face hub."""
        ###########################################################################################
        # TODO (Optional): Please uncomment the following line to add the custom module to the hub model
        # Make sure custom_module is repalced with the name of your custom module class
        # =========================================================================================
        # self.pretrained_model.custom_module = self.custom_module
        ###########################################################################################

        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            **kwargs,
    ):
        """
        Applies a forward pass to the wrapped model and returns the output from the model.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        Returns:
            output_dict (`dict`): A dictionary containing the output from the model.
        """
        kwargs["output_hidden_states"] = True
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        output_dict = {}

        ###############################################################
        # TODO: Please implement your customized forward pass here
        # =============================================================

        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        output_dict = {
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states,
            "past_key_values": outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
        }

        # raise NotImplementedError
        ###############################################################

        return output_dict


    def _get_logprobs(self, model, batch, tokenizer):
        """
        Computes the log probabilities of a response using the model respectively.

        Args:
            batch (`dict` of `list`): A dictionary containing the input data for the DPO model.
                The data format is as follows:
                {
                    "prompt": List[str],
                    "chosen": List[str],
                    "rejected": List[str],
                    "chosen_logps": Optional(torch.FloatTensor)
                    "rejected_logps": Optional(torch.FloatTensor)
                }
            tokenizer (`PreTrainedTokenizerBase`): The tokenizer used to tokenize the input data.
        Returns:
            A tuple of two tensors: (chosen_logps, rejected_logps)
            chosen_logps (`torch.FloatTensor`):
                Log probabilities of the chosen responses. Shape: (batch_size,)
            rejected_logps (`torch.FloatTensor`):
                Log probabilities of the rejected responses. Shape: (batch_size,)
        """

        # https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        # NOTE: This will probably be used for M3, ill leave it here for now
        # Alternative: template: <|user|>\nQuestion <|end|>\n<|assistant|>
        # def tokenize_with_template(prompt, response):
        #         # Apply a template to the prompt
        #         formatted_prompt = f"Question: {prompt}\nAnswer:"
        #         combined_text = f"{formatted_prompt} {response}"
        #         # Tokenize the combined text
        #         return tokenizer(combined_text, padding=False, return_tensors="pt")["input_ids"].to(self.device)

        # prompt_tokens = [tokenizer(f"Question: {p}", padding=False, return_tensors="pt")["input_ids"].to(self.device) for p in batch["prompt"]]
        # chosen_tokens = [tokenize_with_template(p, resp) for p, resp in zip(batch["prompt"], batch["chosen"])]
        # rejected_tokens = [tokenize_with_template(p, resp) for p, resp in zip(batch["prompt"], batch["rejected"])]
        
        # For now I implement this directly by concatenating the tokens from the question and answer for the gpt-2 model
        prompt_tokens = [tokenizer(p, padding=False, return_tensors="pt")["input_ids"].to(self.device) for p in batch["prompt"]]
        chosen_tokens = [tokenizer(resp, padding=False, return_tensors="pt")["input_ids"].to(self.device) for resp in batch["chosen"]]
        rejected_tokens = [tokenizer(resp, padding=False, return_tensors="pt")["input_ids"].to(self.device) for resp in batch["rejected"]]


        def get_logps(prompt_tokens, resp_tokens):
            
            concat_tokens = [torch.cat([prompt_tokens[i], resp_tokens[i]], dim=1)
                             for i in range(len(resp_tokens))]
            maxlen = max([len(t[0]) for t in concat_tokens])

            # For a chat model, we need to only keep the response mask as 1s here
            attention_mask = torch.zeros(len(concat_tokens), maxlen).to(self.device)
            mask = torch.zeros(len(concat_tokens), maxlen).to(self.device)
            for i in range(len(concat_tokens)):
                # Mask out the tokens not corresponding to output
                mask[i, prompt_tokens[i].shape[1]:prompt_tokens[i].shape[1] + resp_tokens[i].shape[1]] = 1
                # Mask out the actual padding
                attention_mask[i, :prompt_tokens[i].shape[1] + resp_tokens[i].shape[1]] = 1

            concat = [torch.cat([t, torch.zeros((1, maxlen - t.shape[1]), dtype=t.dtype).to(self.device)], dim=1) for t in concat_tokens]
            all_tokens = torch.cat(concat, dim=0)

            if all_tokens.shape[1] > self.max_seq_length: 
                all_tokens = all_tokens[:, :self.max_seq_length]
                attention_mask = attention_mask[:, :self.max_seq_length]
                mask = mask[:, :self.max_seq_length]
                print("Warning: Truncated input to max_seq_length")

            outputs = model(input_ids=all_tokens.to(self.device), attention_mask=attention_mask.to(self.device))

            logprobs = F.log_softmax(outputs.logits, dim=-1)
            
            logprobs = torch.gather(logprobs, 2, all_tokens.to(self.device).unsqueeze(-1)).squeeze(-1)

            output = torch.sum(logprobs*mask, dim=-1)

            # clear memory 
            del outputs
            del logprobs
            gc.collect()
            torch.cuda.empty_cache()

            return output

        # TODO: think if we need to normalize this by the length of the output for stability?
        return get_logps(prompt_tokens, chosen_tokens), get_logps(prompt_tokens, rejected_tokens)

    def get_logprobs(self, batch, tokenizer):
        return self._get_logprobs(self.pretrained_model, batch, tokenizer)

    def prediction_step_reward(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
    ):
        """
        Computes the reward socres of the chosen and reject responses by implementing the DPO reward function
        Reference of the DPO reward function: https://arxiv.org/pdf/2305.18290.pdf

        Args:
            policy_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        Returns:
            output_dict (`dict`):
                A dictionary containing the reward scores of the chosen and rejected responses.
        """
        output_dict = {
            "chosen_rewards": [],
            "rejected_rewards": []
        }

        ########################################################################
        # TODO: Please implement the prediction step that computes the rewards
        # ======================================================================

        # From the paper: 
        ## TL;DR summarization, we use β = 0.5, while rest of the parameters remain the same.

        # Just in case we need to calc losses somewhere
        # losses = -F.logsigmoid(temperature * (pi_logratios - ref_logratios))

        # Should we use this for inference?
        temperature = 0.1

        # set logs off device
        policy_chosen_logps = policy_chosen_logps.to("cpu")
        policy_rejected_logps = policy_rejected_logps.to("cpu")
        reference_chosen_logps = reference_chosen_logps.to("cpu")
        reference_rejected_logps = reference_rejected_logps.to("cpu")

        # Calculate rewards
        chosen_rewards = temperature * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = temperature * (policy_rejected_logps - reference_rejected_logps)

        output_dict = {
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": rejected_rewards
        }
        
        # You need to return one reward score for each chosen and rejected response.
        # ======================================================================
        ########################################################################

        return output_dict

    def prediction_step_mcqa(self, batch, tokenizer):
        """
        Computes the mcqa prediction of the given question.

        Args:
            batch (`dict` of `list`):
                A dictionary containing the input mcqa data for the DPO model.
                The data format is as follows:
                {
                    "question": List[str], each <str> contains the question body and the choices
                    "answer": List[str], each <str> is a single letter representing the correct answer
                }
            tokenizer (`PreTrainedTokenizerBase`): The tokenizer used to tokenize the input questions.
        Returns:
            output_dict (`dict`): A dictionary containing the model predictions given input questions.
        """
        output_dict = {"preds": []}

        ########################################################################
        # TODO: Please implement the prediction step that generates the prediction of the given MCQA question
        # ======================================================================

        # HuggingFace pipeline: https://huggingface.co/docs/transformers/main_classes/pipelines. 
        # There exists question-ansewring mode, but i am not sure if Phi-3 supports this.
        ...  # TODO: This might be for Milestone 3! Implement this after discussing with team.

        # You need to return one letter prediction for each question.
        # ======================================================================
        raise NotImplementedError
        ########################################################################

        return output_dict


class AutoDPOModelForSeq2SeqLM(PreTrainedModelWrapper):
    r"""
    A seq2seq model with support for custom modules in addition to the transformer model.
    This class inherits from `~trl.PreTrainedModelWrapper` and wraps a
    `transformers.PreTrainedModel` class. The wrapper class supports classic functions
    such as `from_pretrained` and `push_to_hub` and also provides some additional
    functionalities such as `generate`.

    Args:
        pretrained_model (`transformers.PreTrainedModel`):
            The model to wrap. It should be a causal language model such as GPT2.
            or any model mapped inside the `AutoModelForSeq2SeqLM` class.
        kwargs:
            Additional keyword arguments passed along to any `CustomModule` classes.
    """

    transformers_parent_class = AutoModelForSeq2SeqLM
    lm_head_namings = ["lm_head", "embed_out", "output_projection"]
    ####################################################################################
    # TODO (Optional): Please put any required arguments for your custom module here
    supported_args = ()

    ####################################################################################

    def __init__(self, pretrained_model, **kwargs):
        super().__init__(pretrained_model, **kwargs)
        self.is_encoder_decoder = True
        if not self._has_lm_head():
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        ###########################################################################################
        # TODO (Optional): Please uncomment the following lines to initialize your custom module
        # Make sure CustomModule is repalced with the name of your custom module class
        # Remember that the below lines are just an example
        # You can reanme the class and the variabels to fit your custom module name,
        # just make sure they are consistent in the code
        # =========================================================================================
        # custom_module_kwargs, _, _ = self._split_kwargs(kwargs)
        # self.custom_module = CustomModule(self.pretrained_model.config, **custom_module_kwargs)
        # self._init_weights(**custom_module_kwargs)
        ###########################################################################################

    def _has_lm_head(self):
        # check module names of all modules inside `pretrained_model` to find the language model head
        for name, _module in self.pretrained_model.named_modules():
            if any(attribute in name for attribute in self.lm_head_namings):
                return True
        return False

    def _init_weights(self, **kwargs):
        """
        Initializes the weights of the custom module. The default initialization strategy is random.
        Users can pass a different initialization strategy by passing the `custom_module_init_strategy`
        argument when calling `.from_pretrained`. Supported strategies are:
            - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `CustomModule` class.
        """
        ###############################################################
        # TODO (Optional): Please implement the initialization strategy for your custom module here
        pass
        ###############################################################

    def state_dict(self, *args, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the custom module
        to the state dictionary of the wrapped model by prepending the key with `custom_module.`.

        IMPORTANT: Make sure to replace `custom_module` with the name of your custom module class name.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            pretrained_model_state_dict = {}

        ###########################################################################################
        # TODO (Optional): Please uncomment the following lines to initialize your custom module
        # Make sure "custom_module" is repalced with the name of your custom module class
        # =========================================================================================
        # custom_module_state_dict = self.custom_module.state_dict(*args, **kwargs)
        # for k, v in custom_module_state_dict.items():
        #     pretrained_model_state_dict[f"custom_module.{k}"] = v
        ###########################################################################################
        return pretrained_model_state_dict

    def post_init(self, state_dict):
        r"""
        We add the state dictionary of the custom module to the state dictionary of the wrapped model
        by prepending the key with `custom_module.`. This function removes the `custom_module.` prefix from the
        keys of the custom module state dictionary.

        IMPORTANT: Make sure to replace `custom_module` with the name of your custom module class name.
        """
        if not hasattr(self, 'custom_module'):
            return

        for k in list(state_dict.keys()):
            if "custom_module." in k:
                state_dict[k.replace("custom_module.", "")] = state_dict.pop(k)
        self.custom_module.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                    "cpu" in self.pretrained_model.hf_device_map.values()
                    or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for CustomModule models."
                )

            # get the lm_head device
            for name, module in self.pretrained_model.named_modules():
                if any(attribute in name for attribute in self.lm_head_namings):
                    lm_head_device = module.weight.device
                    break

            # put custom_module on the same device as the lm_head to avoid issues
            self.custom_module = self.custom_module.to(lm_head_device)

            def set_device_hook(module, input, outputs):
                r"""
                A hook that sets the device of the output of the model to the device of the first
                parameter of the model.

                Args:
                    module (`nn.Module`):
                        The module to which the hook is attached.
                    input (`tuple`):
                        The input to the module.
                    outputs (`tuple`):
                        The output of the module.
                """
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(lm_head_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)
            self.is_sequential_parallel = True

    def push_to_hub(self, *args, **kwargs):
        """Push the model to the Hugging Face hub."""
        ###########################################################################################
        # TODO (Optional): Please uncomment the following line to add the custom module to the hub model
        # Make sure custom_module is repalced with the name of your custom module class
        # =========================================================================================
        # self.pretrained_model.custom_module = self.custom_module
        ###########################################################################################

        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the output from the model.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        Returns:
            ouput_dict (`dict`): A dictionary containing the output from the model.
        """
        kwargs["output_hidden_states"] = True
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        ouput_dict = {}

        ###############################################################
        # TODO: Please implement your customized forward pass here
        # =============================================================
        raise NotImplementedError
        ###############################################################

        return ouput_dict

    def get_logprobs(self, batch, tokenizer):
        """
        Computes the log probabilities of a response using the model respectively.

        Args:
            batch (`dict` of `list`): A dictionary containing the input data for the DPO model.
                The data format is as follows:
                {
                    "prompt": List[str],
                    "chosen": List[str],
                    "rejected": List[str],
                    "chosen_logps": Optional(torch.FloatTensor)
                    "rejected_logps": Optional(torch.FloatTensor)
                }
            tokenizer (`PreTrainedTokenizerBase`): The tokenizer used to tokenize the input data.
        Returns:
            A tuple of two tensors: (chosen_logps, rejected_logps)
            chosen_logps (`torch.FloatTensor`):
                Log probabilities of the chosen responses. Shape: (batch_size,)
            rejected_logps (`torch.FloatTensor`):
                Log probabilities of the rejected responses. Shape: (batch_size,)
        """
        ###############################################################
        # TODO: Please implement your customized logprob computation here
        # =============================================================
        raise NotImplementedError
        ###############################################################

        return chosen_logps, rejected_logps

    def prediction_step_reward(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
    ):
        """
        Computes the reward socres of the chosen and reject responses by implementing the DPO reward function
        Reference of the DPO reward function: https://arxiv.org/pdf/2305.18290.pdf

        Args:
            policy_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        Returns:
            output_dict (`dict`):
                A dictionary containing the reward scores of the chosen and rejected responses.
        """
        output_dict = {
            "chosen_rewards": [],
            "rejected_rewards": []
        }

        ########################################################################
        # TODO: Please implement the dpo loss function to compute the rewards
        # You need to return one reward score for each chosen and rejected response.
        # ======================================================================
        raise NotImplementedError
        ########################################################################

        return output_dict

    def prediction_step_mcqa(self, batch, tokenizer):
        """
        Computes the mcqa prediction of the given question.

        Args:
            batch (`dict` of `list`):
                A dictionary containing the input mcqa data for the DPO model.
                The data format is as follows:
                {
                    "question": List[str], each <str> contains the question body and the choices
                    "answer": List[str], each <str> is a single letter representing the correct answer
                }
            tokenizer (`PreTrainedTokenizerBase`): The tokenizer used to tokenize the input questions.
        Returns:
            output_dict (`dict`): A dictionary containing the model predictions given input questions.
        """
        output_dict = {"preds": []}

        ########################################################################
        # TODO: Please implement the prediction step that generates the prediction of the given MCQA question
        # ======================================================================
        # You need to return one letter prediction for each question.
        # ======================================================================
        raise NotImplementedError
        ########################################################################

        return output_dict
