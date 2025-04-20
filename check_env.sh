#!/bin/bash

echo "🔍 Checking Python environment for DPO dependencies..."

check_import () {
  MODULE=$1
  NAME=$2
  python -c "import $MODULE" &> /dev/null
  if [ $? -eq 0 ]; then
    echo "✅ $NAME is installed."
  else
    echo "❌ $NAME is NOT installed."
  fi
}

check_import "torch" "PyTorch"
check_import "transformers" "Transformers"
check_import "datasets" "Datasets"
check_import "accelerate" "Accelerate"
check_import "peft" "PEFT"
check_import "trl" "TRL (DPOTrainer)"
check_import "bitsandbytes" "BitsAndBytes"
check_import "huggingface_hub" "HuggingFace Hub"
check_import "sentencepiece" "SentencePiece"
check_import "protobuf" "Protobuf"
check_import "yaml" "PyYAML"

echo -e "\n🧠 GPU available? ->"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

echo -e "\n✅ Done checking environment!"
