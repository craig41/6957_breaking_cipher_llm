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

python run.py -m  -k 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 -w human_paraphrase_only -d '../../data/full/condaqa_dev.json' -s 0 -n 16
