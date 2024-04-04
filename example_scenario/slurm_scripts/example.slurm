#!/bin/bash
#SBATCH --job-name=example_scenario
#SBATCH --output=outputs/example.out
#SBATCH --err=outputs/example.err
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=defq
#SBATCH --mem=40G
# module load python/3.9 cuda/11.0
source /weka/scratch/tshu2/sye10/miniconda3/etc/profile.d/conda.sh
conda activate habitat2
python /weka/scratch/tshu2/sye10/habitat/example_scenario/testHSSD.py