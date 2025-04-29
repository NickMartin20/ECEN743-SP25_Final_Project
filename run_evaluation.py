import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_from_disk
import evaluate
import pandas as pd
from evaluate.visualization import radar_plot
import os

# ---------------------------
# 1. Load Model + Adapter
# ---------------------------

base_model_path = "/scratch/user/lhuangxu/ECEN743_Final/ECEN743-SP25_Final_Project/llama3_model"    # <<-- CHANGE THIS
adapter_path = "/scratch/user/lhuangxu/ECEN743_Final/ECEN743-SP25_Final_Project/data/DPO_LLAMA"           # <<-- CHANGE THIS

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# ---------------------------
# 2. Load LOCAL Datasets
# ---------------------------

# Load saved MMLU High School Physics
mmlu = load_from_disk("/scratch/user/lhuangxu/ECEN743_Final/ECEN743-SP25_Final_Project/datasets/mmlu_highschool_physics")

# Load saved XSum
xsum = load_from_disk("/scratch/user/lhuangxu/ECEN743_Final/ECEN743-SP25_Final_Project/datasets/xsum_data")

# ---------------------------
# 3. MMLU Evaluation (Multiple Choice)
# ---------------------------

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

predictions_mmlu = []
references_mmlu = []

for example in mmlu['test'].select(range(30)):  # Small sample for demo
    question = example['question']
    choices = example['choices']
    correct_answer = example['answer']

    prompt = f"Question: {question}\nChoices: {', '.join(choices)}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    predicted_letter = output_text.strip()[0].upper()
    predictions_mmlu.append(predicted_letter)
    references_mmlu.append(correct_answer)

# Compute Metrics
acc_mmlu = accuracy_metric.compute(predictions=predictions_mmlu, references=references_mmlu)['accuracy']
f1_mmlu = f1_metric.compute(predictions=predictions_mmlu, references=references_mmlu, average="macro")['f1']

# ---------------------------
# 4. ROUGE Evaluation (Summarization)
# ---------------------------

rouge_metric = evaluate.load("rouge")

predictions_rouge = []
references_rouge = []

for example in xsum['test'].select(range(30)):  # Small sample for demo
    prompt = "Summarize: " + example['document']
    reference = example['summary']

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    predictions_rouge.append(output_text)
    references_rouge.append(reference)

# Compute ROUGE
rouge_scores = rouge_metric.compute(predictions=predictions_rouge, references=references_rouge)

# Pick specific ROUGE scores
rouge1 = rouge_scores['rouge1']
rouge2 = rouge_scores['rouge2']
rougeL = rouge_scores['rougeL']

# ---------------------------
# 5. Save to CSV
# ---------------------------

# Organize results
results_dict = {
    "Accuracy (MMLU Physics)": acc_mmlu,
    "F1 Score (MMLU Physics)": f1_mmlu,
    "ROUGE-1 (XSum Summarization)": rouge1,
    "ROUGE-2 (XSum Summarization)": rouge2,
    "ROUGE-L (XSum Summarization)": rougeL
}
output_dir = "/scratch/user/lhuangxu/ECEN743_Final/ECEN743-SP25_Final_Project/eval_outputs"

import os
output_dir = "/scratch/user/lhuangxu/ECEN743_Final/ECEN743-SP25_Final_Project/eval_outputs_DPO_LLAMA"
os.makedirs(output_dir, exist_ok=True)
print(f"✅ Output folder ready: {output_dir}")

# Save results into CSV
results_csv_path = os.path.join(output_dir, "evaluation_results.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"✅ Saved results to {results_csv_path}")

# Save radar plot
plot_png_path = os.path.join(output_dir, "evaluation_radar_plot.png")
plot.figure.savefig(plot_png_path, dpi=300)
print(f"✅ Radar plot saved to {plot_png_path}")

# Optional: show plot (only if running interactively)
plot.show()
