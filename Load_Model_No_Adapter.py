import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from peft import PeftModel

def format_prompt(input):
    return f"Human: {input}\nAssistant:"

prompts = []
responses = []
#load in csv file
with open('/scratch/user/nmartin20/ECEN743_SP25_Final_Project/prompts/QA_prompts.csv', mode = 'r')as file:
    csvFile = csv.reader(file)
    for lines in csvFile:
        prompts.append(lines[0])

#base model location
base_model_id = '/scratch/user/nmartin20/ECEN743_SP25_Final_Project/pretrained_models/llama3_model'

#load in base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    device_map='auto',
    torch_dtype='auto',
    local_files_only=True
)

print('Base Model Loaded')

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    local_files_only=True
)

print('Tokenizer Loaded')

text_generation_pipeline = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer,
    max_new_tokens = 200,
    truncation=True
)

print("###Base_Model_Response###")
print('\n')
print('\n')

for prompt in prompts:
    response = text_generation_pipeline(format_prompt(prompt))
    print(response)
    print('\n')
    print('\n')
    responses.append(response)