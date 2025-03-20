#!/bin/bash
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --mem=245GB
#SBATCH --mail-user=u1380656@umail.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o outputs-%j

WORKDIR=/scratch/general/vast/$USER/CondaQA_Private
source /uufs/chpc.utah.edu/common/home/u1380656/.venv/bin/activate
module load cuda/12.4.0

nvidia-smi
python -c "import torch; torch.cuda.is_available()"
echo "SLURM JOB ID: $SLURM_JOBID"

mkdir -p /scratch/general/vast/$USER/huggingface_cache
export HF_HOME="/scratch/general/vast/$USER/huggingface_cache"

python run.py -m llama3_3_70B -w translate -d flores
