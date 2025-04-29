# -*- coding: utf-8 -*-

import evaluate

# Try loading BLEU from your local folder
try:
    bleu_metric = evaluate.load("/scratch/user/lhuangxu/ECEN743_Final/ECEN743-SP25_Final_Project/local_metrics/bleu")
    print("[OK] Loaded local BLEU metric successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load local BLEU metric: {e}")
    exit(1)

# Dummy predictions and references
predictions = ["the cat is on the mat", "there is a cat on the mat"]
references = [["the cat is playing on the mat"], ["a cat sits on the mat"]]

# Try computing BLEU
try:
    bleu_result = bleu_metric.compute(predictions=predictions, references=references)
    print(f"[OK] BLEU score computed: {bleu_result['bleu']:.4f}")
except Exception as e:
    print(f"[ERROR] Failed to compute BLEU score: {e}")
    exit(1)
