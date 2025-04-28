#!/bin/bash
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mem=100GB
#SBATCH --mail-user=u1380656@umail.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o outputs/outputs-%j

EPOCHS=3
if [ "$#" -ge 1 ]; then
    EPOCHS=$1
fi
# Environment setup
module load cuda/12.4.0

source ~/.bashrc
source /scratch/general/vast/u1380656/6957_breaking_cipher_llm/.venv/bin/activate

echo "SLURM JOB ID: $SLURM_JOBID"
echo "Training for $EPOCHS epochs"

# Set up cache directory
mkdir -p /scratch/general/vast/$USER/huggingface_cache
export HF_HOME="/scratch/general/vast/$USER/huggingface_cache"

python train_lora_aya.py --epochs $EPOCHS