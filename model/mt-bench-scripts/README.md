## Evaluate LLM on MT-bench. Repo: https://github.com/lm-sys/FastChat

### Steps

FastChat uses the `Conversation` class to handle prompt templates and `BaseModelAdapter` class to handle model loading.

1. Implement a conversation template for the new model at [fastchat/conversation.py](https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py). You can follow existing examples and use `register_conv_template` to add a new one. Please also add a link to the official reference code if possible.

The template we use: 
```
#Phi-3-mini
#reference: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
register_conv_template(
    Conversation(
        name="phi-3",
        system_message="",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="<end>",
    )
)
```
2. Place the mt-bench-scripts in /FastChat/fastchat/llm_judge folder and run them to reproduce our results.
3. For experimentation you can fork this repo: https://github.com/Stefanstud/FastChat
