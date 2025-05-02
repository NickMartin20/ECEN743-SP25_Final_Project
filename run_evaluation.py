
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_from_disk
import evaluate
import pandas as pd
from evaluate.visualization import radar_plot
import os
from datetime import datetime

# Safe matplotlib import
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] matplotlib not found! Please install it. Exiting...")
    exit(1)

from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------
# 0. Helper: Check Metrics Cached
# ---------------------------

def check_metrics_cached(metrics_list):
    missing_metrics = []
    for metric_name in metrics_list:
        try:
            _ = evaluate.load(metric_name)
            print(f"[OK] Metric '{metric_name}' cached and ready.")
        except Exception as e:
            print(f"[ERROR] Metric '{metric_name}' missing! Error: {e}")
            missing_metrics.append(metric_name)
    return missing_metrics

metrics_to_check = ["accuracy", "f1", "rouge"]
missing = check_metrics_cached(metrics_to_check)
if missing:
    print("\n[EXIT] Some metrics are missing.")
    exit(1)

# ---------------------------
# 1. Load Model + Adapter
# ---------------------------

base_model_path = "/scratch/user/lhuangxu/ECEN743_Final/ECEN743-SP25_Final_Project/llama3_model"
adapter_path = "/scratch/user/lhuangxu/ECEN743_Final/ECEN743-SP25_Final_Project/data/DPO_LLAMA_HUMANLY"

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)
model = PeftModel.from_pretrained(model, adapter_path)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    local_files_only=True
)

# ---------------------------
# 2. Load LOCAL Datasets
# ---------------------------

mmlu = load_from_disk("/scratch/user/lhuangxu/ECEN743_Final/ECEN743-SP25_Final_Project/datasets/mmlu_highschool_physics")
xsum = load_from_disk("/scratch/user/lhuangxu/ECEN743_Final/ECEN743-SP25_Final_Project/datasets/xsum_data")

# ---------------------------
# 3. Load Metrics
# ---------------------------

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
rouge_metric = evaluate.load("rouge")

# ---------------------------
# 4. MMLU Evaluation (Multiple Choice)
# ---------------------------

label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
default_guess = 0  # Guess "A" if invalid

predictions_mmlu = []
references_mmlu = []

for example in mmlu['test'].select(range(30)):  # Small sample
    question = example['question']
    choices = example['choices']
    correct_answer = example['answer']

    try:
        correct_idx = int(correct_answer)
    except ValueError:
        correct_idx = -1

    prompt = (
        f"Question: {question}\n"
        f"Choices:\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n\n"
        f"Please answer with only the letter corresponding to the correct answer (A, B, C, or D)."
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    predicted_letter = output_text.strip()[0].upper()

    if predicted_letter in label_map:
        predictions_mmlu.append(label_map[predicted_letter])
    else:
        print(f"[WARN] Invalid model output '{predicted_letter}', defaulting to 'A'")
        predictions_mmlu.append(default_guess)

    if correct_idx in [0,1,2,3]:
        references_mmlu.append(correct_idx)
    else:
        print(f"[WARN] Invalid reference '{correct_answer}', defaulting to 0")
        references_mmlu.append(0)

# Compute Accuracy
acc_mmlu = accuracy_metric.compute(predictions=predictions_mmlu, references=references_mmlu)['accuracy']

# Compute F1 safely
try:
    f1_mmlu = f1_metric.compute(predictions=predictions_mmlu, references=references_mmlu, average="macro")['f1']
except Exception as e:
    print(f"[WARN] F1 computation failed: {e}")
    f1_mmlu = 0.0

# Confusion Matrix
cm = confusion_matrix(references_mmlu, predictions_mmlu, labels=[0,1,2,3])
print("\nConfusion Matrix (0=A, 1=B, 2=C, 3=D):\n", cm)

# Per-Class Report
report = classification_report(references_mmlu, predictions_mmlu, labels=[0,1,2,3], output_dict=True)
print("\nPer-Class Report:")
for label_idx, label_name in zip([0,1,2,3], ["A", "B", "C", "D"]):
    print(f"Class {label_name}: Precision={report[str(label_idx)]['precision']:.2f}, Recall={report[str(label_idx)]['recall']:.2f}, F1={report[str(label_idx)]['f1-score']:.2f}")

# ---------------------------
# 5. ROUGE Evaluation (Summarization)
# ---------------------------

predictions_rouge = []
references_rouge = []

for example in xsum['test'].select(range(30)):  # Small sample
    prompt = "Summarize: " + example['document']
    reference = example['summary']

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    predictions_rouge.append(output_text)
    references_rouge.append(reference)

rouge_scores = rouge_metric.compute(predictions=predictions_rouge, references=references_rouge)
rouge1 = rouge_scores['rouge1']
rouge2 = rouge_scores['rouge2']
rougeL = rouge_scores['rougeL']

# ---------------------------
# 6. Save Results and Outputs
# ---------------------------

# Timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"/scratch/user/lhuangxu/ECEN743_Final/ECEN743-SP25_Final_Project/eval_outputs_DPO_LLAMA"
os.makedirs(output_dir, exist_ok=True)
print(f"[OK] Output folder ready: {output_dir}")

# Save CSV
csv_filename = f"evaluation_results_{timestamp}.csv"
results_csv_path = os.path.join(output_dir, csv_filename)
results_dict = {
    "Accuracy (MMLU Physics)": acc_mmlu,
    "F1 Score (MMLU Physics)": f1_mmlu,
    "ROUGE-1 (XSum Summarization)": rouge1,
    "ROUGE-2 (XSum Summarization)": rouge2,
    "ROUGE-L (XSum Summarization)": rougeL
}
results_df = pd.DataFrame([results_dict])
results_df.to_csv(results_csv_path, index=False)
print(f"[OK] Saved CSV to {results_csv_path}")

# Save radar plot
plot = radar_plot(
    data=[results_dict],
    model_names=["My-LLM-Adapter"],
)

plot_png_path = os.path.join(output_dir, "evaluation_radar_plot.png")
plot.figure.savefig(plot_png_path, dpi=300)
print(f"[OK] Saved radar plot to {plot_png_path}")


