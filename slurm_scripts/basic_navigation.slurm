#!/bin/bash
#SBATCH --job-name=habitat_sim
#SBATCH --output=/weka/scratch/tshu2/sye10/slurm_scripts/outputs/habitat_lab_%j.out
#SBATCH --error=/weka/scratch/tshu2/sye10/slurm_scripts/outputs/habitat_lab_%j.err
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=defq
#SBATCH --mem=8G

# module load python/3.9 cuda/11.0

