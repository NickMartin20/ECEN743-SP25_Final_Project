import os
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = '/scratch/user/nmartin20/ECEN743_SP25_Final_Project/pretrained_models/llama3-model'
print("Files in model dir:", os.listdir(repo_id))

model = AutoModelForCausalLM.from_pretrained(
    repo_id, 
    device_map="cuda:0",
    local_file_only=True
)

tokenizer = AutoTokenizer.from_pretrained(
    repo_id,
    local_files_only=True
)

print("Model and tokenizer loaded locally.")