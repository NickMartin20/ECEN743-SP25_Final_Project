#!/bin/bash
#SBATCH --job-name=RunSFT
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --output=%j.log


#SBATCH --mail-type=ALL
#SBATCH --mail-user=nmartin20@tamu.edu


ml purge
ml GCCcore/14.2.0
ml Miniconda3/23.10.0-1


source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate rl_env

cd $SCRATCH/ECEN743-SP25-PG

echo "Starting Main!!!"
echo "Allocated GPU(s):"
nvidia-smi
echo "Loaded modules:"
module list

python SFT_Model.py

echo "Done!!!"