# Run this to generate answers for the model
# Note: Generating all responses can take a while, ~120 minutes or more.

# First for the base model Phi-3-mini-4k
python3 gen_model_answer.py --model-id "phi-3" --model-path "microsoft/Phi-3-mini-4k-instruct"

# Then generate for the DPO model
python3 gen_model_answer.py --model-id "phi-3-dpo" --model-path "StefanKrsteski/Phi-3-mini-4k-instruct-DPO-EPFL"