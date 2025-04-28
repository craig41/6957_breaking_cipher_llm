#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=100GB
#SBATCH --mail-user=u1380656@umail.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o outputs-%j

WORKDIR=/scratch/general/vast/$USER/6957_breaking_cipher_llm
source $WORKDIR/.venv/bin/activate
module load cuda/12.4.0

nvidia-smi
python -c "import torch; torch.cuda.is_available()"
echo "SLURM JOB ID: $SLURM_JOBID"

mkdir -p /scratch/general/vast/$USER/huggingface_cache
export HF_HOME="/scratch/general/vast/$USER/huggingface_cache"

# Define base directories
GOLD_DIR="$WORKDIR/data/gold"
TRANSLATIONS_DIR="$WORKDIR/data/lora_full"
OUTPUT_DIR="$WORKDIR/eval/bert_score_results"

# Define paths for this evaluation
GOLD_PATH="$GOLD_DIR/translated_100.txt"
TRANS_PATH="$TRANSLATIONS_DIR/llama_base_small/translated.txt"
OUTPUT_PATH="$TRANSLATIONS_DIR/llama_base_small/bert_results.txt"

# Run the BERT evaluation
python $WORKDIR/eval/bert_eval.py \
    --gold "$GOLD_PATH" \
    --translations "$TRANS_PATH" \
    --output "$OUTPUT_PATH"

echo "Results saved to $OUTPUT_PATH"
