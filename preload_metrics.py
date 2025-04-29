# preload_metrics.py

import evaluate
import os

os.environ["HF_MODULES_CACHE"] = "/scratch/user/lhuangxu/huggingface_cache"

for metric_name in ["accuracy", "f1", "rouge", "bleu"]:
    print(f"Loading metric: {metric_name}")
    metric = evaluate.load(metric_name, download_mode="reuse_cache_if_exists")
