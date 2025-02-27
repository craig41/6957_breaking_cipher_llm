#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=80GB
#SBATCH --mail-user=u1380656@umail.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o outputs-%j

WORKDIR=/scratch/general/vast/$USER/CondaQA_Private
source /scratch/general/vast/$USER/CondaQA_Private/.venv/bin/activate

python run.py -m llama3_3_70B -w translate -d flores
