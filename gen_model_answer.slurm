#!/bin/bash
#SBATCH --job-name=gen_answers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --output=gen_answers.out
#SBATCH --error=gen_answers.err
set -e
module load CUDA/12.2.2
module load Miniconda3/23.10.0-1
source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate rl_env

export PATH=/scratch/user/lhuangxu/.local/bin:$PATH
export PYTHONPATH=/scratch/user/lhuangxu/.local/lib/python3.11/site-packages:$PYTHONPATH
export PATH=/scratch/user/lhuangxu/.local/bin:$PATH
export PYTHONPATH=/scratch/user/lhuangxu/.local/lib/python3.11/site-packages:$PYTHONPATH
cd /scratch/user/lhuangxu/ECEN743_Final/ECEN743-SP25_Final_Project
python3 -m fastchat.llm_judge.gen_model_answer \
  --model-path /scratch/user/lhuangxu/ECEN743_Final/ECEN743-SP25_Final_Project/data/DPO_LLAMA \
  --model-id llama3_dpo \
  --answer-file /scratch/user/lhuangxu/ECEN743_Final/ECEN743-SP25_Final_Project/llama3_dpo_answers.json \
  --num-gpus-per-model 1 \
  --num-gpus-total 2 \
  --max-new-token 512 \
  --dtype float16
  --question-end 40