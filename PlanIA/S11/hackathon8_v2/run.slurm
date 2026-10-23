#!/bin/bash
#SBATCH --job-name=h8-qwen35
#SBATCH --account=investigacion2
#SBATCH --partition=gpu
#SBATCH --nodelist=ds001
#SBATCH --gres=shard:rtxa6000:7
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/fabryzzio.meza/hackathon8_qwen35/logs/experiment-%j.out
#SBATCH --error=/home/fabryzzio.meza/hackathon8_qwen35/logs/experiment-%j.err
set -euo pipefail
cd /home/fabryzzio.meza/hackathon8_qwen35
test -f SETUP_READY
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export CPATH="$PWD/compiler_include${CPATH:+:$CPATH}"
nvidia-smi
# Python hereda los recursos GPU asignados al trabajo de Slurm.
# Los shards no se vuelven a solicitar en un paso srun.
./venv/bin/python src/experiment.py
