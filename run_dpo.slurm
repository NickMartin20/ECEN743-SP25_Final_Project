#!/bin/bash
#SBATCH --job-name=DPO_LLAMA_HUMANLY
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --output=%j_dpo.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lhuangxu@tamu.edu

# Load environment modules
module purge
module load CUDA/12.2.2
module load Miniconda3/23.10.0-1
echo "CUDA_HOME: $CUDA_HOME"

# Initialize and activate conda
source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate rl_env

# Change to working directory
cd /scratch/user/lhuangxu/ECEN743_Final/ECEN743-SP25_Final_Project

# Confirm correct python and torch
echo "PYTHON PATH: $CONDA_PREFIX/bin/python"
$CONDA_PREFIX/bin/python -c "import torch; print(torch.__version__)"

echo "====== Checking Python imports before training ======"
$CONDA_PREFIX/bin/python check_imports.py
if [ $? -ne 0 ]; then
  echo "Missing imports detected. Exiting."
  exit 1
fi
echo "====== Imports OK. Continuing to training ======"


echo "====== Starting DPO Fine-Tuning ======"
echo "Allocated GPU(s):"
nvidia-smi
echo "Loaded modules:"
module list
echo "Current conda environment: $CONDA_DEFAULT_ENV"

# Change to working directory
cd /scratch/user/lhuangxu/ECEN743_Final/ECEN743-SP25_Final_Project

# Run the DPO script
PYTHONPATH=/scratch/user/lhuangxu/ECEN743_Final/alignment-handbook \
$CONDA_PREFIX/bin/python run_dpo.py config_openhermes.yaml

echo "====== DPO Training Complete ======"
