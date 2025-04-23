#!/bin/bash

#SBATCH -J SpeculativeDecoding
#SBATCH --account=gpuperfportability
#SBATCH --partition=l40s_dev_q
#SBATCH --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1 --gres-flags=enforce-binding
#SBATCH --output=slurm_logs/SpecDec_%j_%x_%N.out
#SBATCH --error=slurm_logs/SpecDec_%j_%x_%N.err

module reset
module load Anaconda3/2020.11

source activate
source deactivate

conda run -n transformers_env python main.py --prompt "Emily found a mysterious letter on her doorstep one sunny morning."
