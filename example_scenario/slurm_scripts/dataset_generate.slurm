#!/bin/bash
#SBATCH --job-name=dataset_generation
#SBATCH --output=outputs/dataset_generation.out
#SBATCH --err=outputs/dataset_generation.err
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=defq
#SBATCH --mem=40G
# module load python/3.9 cuda/11.0
source /weka/scratch/tshu2/hshi33/miniconda/etc/profile.d/conda.sh
conda activate habitat
python -m habitat.datasets.rearrange.run_episode_generator --run --config /weka/scratch/tshu2/hshi33/habitat/example_scenario/configuration/dataset.yaml --num-episodes 10 --out /weka/scratch/tshu2/hshi33/habitat/example_scenario/dataset.json.gz