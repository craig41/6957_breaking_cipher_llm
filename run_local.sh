#!/bin/bash
#SBATCH --job-name=final_llama_eval
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=100GB
#SBATCH --mail-user=u0013114@umail.utah.edu
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
TEST_DATA="/uufs/chpc.utah.edu/common/home/u0013114/code/6957_breaking_cipher_llm/data/unused_data/final_test/final_eval.csv"
# LORA_PATH="/scratch/general/vast/u1380656/6957_breaking_cipher_llm/src/results_llama3-8b-instruct-translator/cipher_adapter-new"
LORA_PATH=/scratch/general/vast/u1380656/6957_breaking_cipher_llm/src/results_aya8b_translator

pip install models

# # Run with LoRA adapter on test CSV data
python run_local.py \
    -m llama3_1_8b_instruct_lora \
    -w translate \
    -n 200 \
    -k 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 \
    --output /uufs/chpc.utah.edu/common/home/u0013114/code/6957_breaking_cipher_llm/data/final_eval_pred/ \
    --lora-path $LORA_PATH \
    --csv-data $TEST_DATA

# # Run with LoRA adapter on test CSV data
# python run_local.py \
#     -m aya_expanse_8b_lora \
#     -w translate \
#     -n 200 \
#     -k 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 \
#     --output /uufs/chpc.utah.edu/common/home/u0013114/code/6957_breaking_cipher_llm/data/final_eval_pred/ \
#     --lora-path $LORA_PATH \
#     --csv-data $TEST_DATA

# Run with standard model for comparison (optional)
# python run_local.py \
#     -m llama3_1_8b_instruct \
#     -w translate \
#     -n 200 \
#     -k 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 \
#     --output /uufs/chpc.utah.edu/common/home/u0013114/code/6957_breaking_cipher_llm/data/final_eval_pred/base_llama/ \
#     --csv-data $TEST_DATA

# Run with standard model for comparison (optional)
# python run_local.py \
#     -m aya_expanse_8b \
#     -w translate \
#     -n 200 \
#     -k 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 \
#     --output /uufs/chpc.utah.edu/common/home/u0013114/code/6957_breaking_cipher_llm/data/final_eval_pred/base_aya/ \
#     --csv-data $TEST_DATA