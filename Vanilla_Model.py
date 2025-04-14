import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

access_token = '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'
model_id = "meta-llama/Llama-3.2-1B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_id)

chat_history_ids = None

for step in range(5):
    user_input = input(">> User: ")

    new_input = tokenizer(user_input, return_tensors='pt', add_special_tokens=True)
    input_ids = new_input["input_ids"]
    attention_mask = new_input["attention_mask"]

    # Append to chat history if it exists
    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids, input_ids], dim=-1)
        attention_mask = torch.cat([torch.ones_like(chat_history_ids), attention_mask], dim=-1)

    # Generate response
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=input_ids.shape[-1] + 100,
        pad_token_id=tokenizer.eos_token_id
    )

    # Slice new tokens and decode
    response = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print("############################")
    print("LlaMA-3.2-1B:", response)

    # Update chat history
    chat_history_ids = output_ids