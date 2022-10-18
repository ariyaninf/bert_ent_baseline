#!/bin/bash
#SBATCH --job-name=train_cl
#SBATCH --error=e.%x.%j
#SBATCH --output=o.%x.%j
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --account=scw1997

set -eu

module purge
module load anaconda
module list

source activate
source activate myenv

WORKDIR=/scratch/$USER/NEW_bert-baseline
cd ${WORKDIR}

python3 main.py
