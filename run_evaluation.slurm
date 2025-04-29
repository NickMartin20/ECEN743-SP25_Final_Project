#!/bin/bash
#SBATCH --job-name=EVAL_LLAMA_ADAPTER
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --output=%j_eval.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lhuangxu@tamu.edu


module purge
module load CUDA/12.2.2
module load Miniconda3/23.10.0-1
echo "CUDA_HOME: $CUDA_HOME"


source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate rl_env


echo "PYTHON PATH: $CONDA_PREFIX/bin/python"
$CONDA_PREFIX/bin/python -c "import torch; print(torch.__version__)"

echo "====== Starting LLM Evaluation (MMLU + ROUGE) ======"
echo "Allocated GPU(s):"
nvidia-smi
echo "Loaded modules:"
module list
echo "Current conda environment: $CONDA_DEFAULT_ENV"


cd /scratch/user/lhuangxu/ECEN743_Final/ECEN743-SP25_Final_Project


$CONDA_PREFIX/bin/python run_evaluation.py

echo "====== LLM Evaluation Complete ======"
