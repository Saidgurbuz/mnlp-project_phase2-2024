# Run this to generate judgments for the models, two modes, either single or pairwise
# NOTE: You need to have openai key for gpt-4

export OPENAI_API_KEY=XXXXX # set the OpenAI API key

# Generate pairwise judgment 
# python gen_judgment.py --model-list phi-3 phi-3-dpo --mode "pairwise-all" 

# Generate single
python gen_judgment.py --model-list phi-3 phi-3-dpo
