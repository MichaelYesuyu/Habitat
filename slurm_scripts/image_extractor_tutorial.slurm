#!/bin/bash
#SBATCH --job-name=habitat_sim
#SBATCH --output=/weka/scratch/tshu2/sye10/habitat/slurm_scripts/outputs/habitat_lab_%j.out
#SBATCH --error=/weka/scratch/tshu2/sye10/habitat/slurm_scripts/outputs/habitat_lab_%j.err
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=defq
#SBATCH --mem=8G

# module load python/3.9 cuda/11.0

source /weka/scratch/tshu2/sye10/miniconda3/etc/profile.d/conda.sh
conda activate habitat2

nvidia-smi

python /weka/scratch/tshu2/sye10/habitat/python_scripts/image_extractor_tutorial.py