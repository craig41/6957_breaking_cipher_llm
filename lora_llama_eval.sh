#!/bin/bash

#SBATCH --job-name=lora_llama_eval
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --mail-user=u0013114@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --output=outputs-%j
#SBATCH --nodes=1
#SBATCH --mem=250GB
#SBATCH --ntasks=1
#SBATCH --time=5:00:00

#SBATCH -o slurmjob-%j.out-%N
#SBATCH -e slurmjob-%j.err-%N

export LD_LIBRARY_PATH=/usr/local/cuda/lib64

WORKDIR=/scratch/general/vast/$USER/CondaQA_Private
OUTDIR=$WORKDIR/src/eval/chat-eval/llama3-3-70B/gen_edits_gen_qs/
# Path to the LoRA adapter
LORA_PATH=/scratch/general/vast/u1380656/6957_breaking_cipher_llm/src/results_llama3-8b-instruct-translator/cipher_adapter-final
module load cuda/12.4.0

echo $OUTDIR

nvidia-smi
python -c "import torch; torch.cuda.is_available()"
echo "SLURM JOB ID: $SLURM_JOBID"

mkdir -p /scratch/general/vast/$USER/huggingface_cache
mkdir -p $OUTDIR
export HF_HOME="/scratch/general/vast/$USER/huggingface_cache"

pip install 'accelerate>=0.26.0'
pip install transformers
pip install peft

python3 lora_llama_eval.py --lora_path LORA_PATH