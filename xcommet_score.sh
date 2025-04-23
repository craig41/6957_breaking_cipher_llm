#!/bin/bash

#SBATCH --job-name=xcommet_score
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --gres=gpu
#SBATCH --requeue
#SBATCH --mail-user=u0013114@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --output=outputs-%j
#SBATCH --nodes=1
#SBATCH --mem=245GB
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

#SBATCH -o slurmjob-%j.out-%N
#SBATCH -e slurmjob-%j.err-%N

export LD_LIBRARY_PATH=/usr/local/cuda/lib64

WORKDIR=/uufs/chpc.utah.edu/common/home/u0013114/code/6957_breaking_cipher_llm
OUTDIR=/uufs/chpc.utah.edu/common/home/u0013114/code/6957_breaking_cipher_llm
module load cuda/12.4.0

nvidia-smi
python -c "import torch; torch.cuda.is_available()"
echo "SLURM JOB ID: $SLURM_JOBID"

mkdir -p /scratch/general/vast/$USER/huggingface_cache
mkdir -p $OUTDIR
export HF_HOME="/scratch/general/vast/$USER/huggingface_cache"

pip install unbabel-comet
python3 xcommet_score.py