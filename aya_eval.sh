#!/bin/bash

#SBATCH --job-name=aya_eval
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
#SBATCH --time=5:00:00

#SBATCH -o slurmjob-%j.out-%N
#SBATCH -e slurmjob-%j.err-%N

export LD_LIBRARY_PATH=/usr/local/cuda/lib64

WORKDIR=/scratch/general/vast/$USER/CondaQA_Private
OUTDIR=$WORKDIR/src/eval/chat-eval/llama3-3-70B/gen_edits_gen_qs/
source /uufs/chpc.utah.edu/common/home/u0013114/code/6957_braeking_cipher_llm
module load cuda/12.4.0

nvidia-smi
python -c "import torch; torch.cuda.is_available()"
echo "SLURM JOB ID: $SLURM_JOBID"

mkdir -p /scratch/general/vast/$USER/huggingface_cache
mkdir -p $OUTDIR
export HF_HOME="/scratch/general/vast/$USER/huggingface_cache"

python3 aya_eval.py --output_dir $OUTDIR