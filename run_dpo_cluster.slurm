#!/bin/bash
#SBATCH --job-name=DPO_FineLLama
#SBATCH --time=08:00:00                       # Adjust depending on dataset size and model
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8                   # Matches your CPUs to help with tokenization
#SBATCH --gres=gpu:a100:1                     # 1 A100 GPU
#SBATCH --mem=80G                             # 80GB for larger models and preprocessing
#SBATCH --output=%j_dpo.log                   # Job ID log file
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nmartin20@tamu.edu

# Load environment modules
ml purge
ml GCCcore/14.2.0
ml Miniconda3/23.10.0-1

# Activate conda environment with your RLHF stack
source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate rl_env

# Change to your scratch project directory
cd $SCRATCH/ECEN743-SP25-PG

echo "====== Starting DPO Fine-Tuning ======"
echo "Allocated GPU(s):"
nvidia-smi
echo "Loaded modules:"
module list
echo "Current conda environment: $CONDA_DEFAULT_ENV"

# Run the DPO script with config
python run_dpo.py config_finely_tuned.yaml

echo "====== DPO Training Complete ======"
