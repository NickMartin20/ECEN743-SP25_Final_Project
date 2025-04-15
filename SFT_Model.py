import os
import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import hf_hub_download
from trl import SFTConfig, SFTTrainer

print('Imports Complete')

####### Loading in the model #######
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True, 
    bnb_8bit_quant_type="nf4", 
    bnb_8bit_use_double_quant=True, 
    bnb_8pipbit_compute_dtype=torch.float32
)

print('Bits and Bytes Config Complete')

#change this repo_id to the dir of your model
repo_id = '/scratch/user/nmartin20/ECEN743_SP25_Final_Project/pretrained_models/llama3-model'
model = AutoModelForCausalLM.from_pretrained(
    repo_id, device_map="cuda:0", quantization_config=bnb_config, local_files_only=True
)
###DEBUG###
#Shows how much space the model occupies in memory
print('MODEL LOADED')
print(model.get_memory_footprint()/1e6)
###########

####### Low-Rank Adapters (LoRA) #######
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    #Lower the ranker, Fewer parameters to train
    r=8,
    lora_alpha=16,
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM", 

    target_modules=['o_proj','qkv_proj','gate_up_proj','down_proj']
)


###DEBUG###
#Printing various parameters
#model = get_peft_model(model, config)
train_p, tot_p = model.model.get_nb_trainable_parameters()
print(f'Trainable parameters:          {train_p/1e6:.2f}M')
print(f'Total Parameters:              {tot_p/1e6:.2f}M')
print(f'% of trainable parameters:     {100*train_p/tot_p:.2f}%')
###########

####### preparing the data #######
'''
Pay attention to the format of the dataset:
conversational format: supported by the trl package
instruction format: NOT support by the trl package

If data set is in instruction format you will have to 
convert it to the conversational format.
'''

#Code for dataset already in conversational format
dataset_id =''
dataset = load_dataset(dataset_id, split="train")

####### Loading the Tokenizer #######
tokenizer = AutoTokenizer.from_pretrained(repo_id, local_files_only=True) #load the tokenizer: converts words and letters to tokens

#THESE LINES MAY POTENTIAL CAUSE ISSUES FOR LLAMA-3.2 if EOS token is masked in the labels of the dataset
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.unk_token_id

###DEBUG###
#print(tokenizer.apply_chat_template(messages, tokenize=False))
###########

####### Fine-Tuning the LLM #######
'''
SFTTrainer works if we have:
1.) a model
2.) a tokenizer
3.) a dataset
4.) a configuration object (We create that here)

'''

sft_config = SFTConfig(
    #Group 1: Memory Usage
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    gradient_accumulation_steps  = 1,
    per_device_train_batch_size = 16,
    auto_find_batch_size=True,

    #Group 2: Dataset-related
    max_seq_length=64,
    packing=True,

    #Group 3: Training Parameters
    num_train_epochs=10,
    learning_rate=3e-4,
    optim='paged_adamw_8bit',

    #Group 4: Logging parameters
    logging_steps=10,
    logging_dir='./logs',
    output_dir='./Llama-3.2-1B-SFT',
    report_to='none'
)

#Now we can train by calling
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_config,
    train_dataset=dataset
)
