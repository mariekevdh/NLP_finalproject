#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --mem=4000

module purge
module load Python/3.11.3-GCCcore-12.3.0

python3 -m venv $HOME/venvs/nlp_fp

source $HOME/venvs/nlp_fp/bin/activate

python3 finetune_model.py -qe 'mqm' -th 0.6
