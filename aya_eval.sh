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

export LD_LIBRARY_PATH=/usr/local/cuda/lib64

WORKDIR=/scratch/general/vast/$USER/CondaQA_Private
OUTDIR=$WORKDIR/src/eval/chat-eval/llama3-3-70B/gen_edits_gen_qs/
#source /uufs/chpc.utah.edu/common/home/u0013114/.venv/bin/activate
module load cuda/12.4.0

nvidia-smi
python -c "import torch; torch.cuda.is_available()"
echo "SLURM JOB ID: $SLURM_JOBID"

mkdir -p /scratch/general/vast/$USER/huggingface_cache
mkdir -p $OUTDIR
export HF_HOME="/scratch/general/vast/$USER/huggingface_cache"


#python condaqa_eval.py --test_file $WORKDIR/data/final_condaqa_labeled/gen_edits_gen_qs_full.json --output_dir $OUTDIR --model_name llama3_3_70b --model_type huggingface --hk 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
python3 aya_eval.py