#!/bin/bash
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
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

# Path to test data
TEST_DATA="$WORKDIR/data/finetune-data-full/combined_test_data.csv"
LORA_PATH="$WORKDIR/src/results_aya8b_translator/system_prompt"

# Run with LoRA adapter on test CSV data
python run_local.py \
    -m aya_expanse_8b \
    -w translate \
    -n 100 \
    -k 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 \
    --output $WORKDIR/data/lora_full/aya_expanse_base_small/  \
    --csv-data $TEST_DATA

# Run with standard model for comparison (optional)
# python run_local.py \
#     -m llama3_1_8b_instruct \
#     -w translate \
#     -n 20 \
#     -k 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 \
#     --output $WORKDIR/data/lora_test/standard_model/ \
#     --csv-data $TEST_DATA