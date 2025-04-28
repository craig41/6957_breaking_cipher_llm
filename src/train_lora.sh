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

# Default number of epochs if not provided
EPOCHS=10
if [ "$#" -ge 1 ]; then
    EPOCHS=$1
fi

WORKDIR=/scratch/general/vast/$USER/6957_breaking_cipher_llm
cd $WORKDIR/src

# Load modules and activate virtual environment
source $WORKDIR/.venv/bin/activate
module load cuda/12.4.0

# Print environment info
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Training for $EPOCHS epochs"

# Set up cache directory
mkdir -p /scratch/general/vast/$USER/huggingface_cache
export HF_HOME="/scratch/general/vast/$USER/huggingface_cache"

# Run the Python script with the specified number of epochs
python train_lora_correct.py --epochs $EPOCHS

echo "Training completed with $EPOCHS epochs"