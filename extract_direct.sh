#!/bin/bash
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=6:00:00
#SBATCH --mem=100GB
#SBATCH --mail-user=u1380656@umail.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o outputs/outputs-%j

WORKDIR=/scratch/general/vast/$USER/6957_breaking_cipher_llm
source $WORKDIR/.venv/bin/activate
module load cuda/12.4.0

nvidia-smi
python -c "import torch; torch.cuda.is_available()"
echo "SLURM JOB ID: $SLURM_JOBID"

mkdir -p /scratch/general/vast/$USER/huggingface_cache
export HF_HOME="/scratch/general/vast/$USER/huggingface_cache"

# Define variables
INPUT_FILE="$WORKDIR/src/outputs/outputs-4125009"
OUTPUT_FILE="$WORKDIR/data/lora_full/aya_expanse_8b/parsed_translations.txt"
GOLD_FILE="$WORKDIR/data/gold/original.txt"
BATCH_SIZE=16  # Process this many translations in a batch

# Install any needed packages
pip install tqdm

# Run the extraction script with direct model loading
python $WORKDIR/extract_direct_translations.py \
    "$INPUT_FILE" \
    "$OUTPUT_FILE" \
    "$GOLD_FILE" \
    $BATCH_SIZE

# Verify the output has the correct number of lines
WC_OUTPUT=$(wc -l "$OUTPUT_FILE")
echo "Output file line count: $WC_OUTPUT"
WC_GOLD=$(wc -l "$GOLD_FILE")
echo "Gold file line count: $WC_GOLD"

# Show the first few lines of the output
echo "First 10 lines of output:"
head -n 10 "$OUTPUT_FILE"