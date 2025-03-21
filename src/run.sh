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
source /uufs/chpc.utah.edu/common/home/u1380656/.venv/bin/activate
module load cuda/12.4.0

nvidia-smi
python -c "import torch; torch.cuda.is_available()"
echo "SLURM JOB ID: $SLURM_JOBID"

mkdir -p /scratch/general/vast/$USER/huggingface_cache
export HF_HOME="/scratch/general/vast/$USER/huggingface_cache"

python run.py -m aya_expanse_8b -w translate -d openlanguagedata/flores_plus -n 200 -k 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 -src kach1280 --output $WORKDIR/data/aya_expanse8b/jingpho/
