#!/bin/bash
#SBATCH --job-name=qwen-test
#SBATCH --account=pregrado
#SBATCH --partition=gpu
#SBATCH --gres=shard:rtxa6000:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-qwen-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

printf 'Job %s | Nodo %s | Inicio %s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date --iso-8601=seconds)"

# El proceso batch ya está dentro de la asignación GPU de Slurm.
# Los shards comparten la VRAM del dispositivo; no reservan VRAM aislada.
.venv/bin/python -u qwen_test.py
printf 'Finalizado: %s\n' "$(date --iso-8601=seconds)"
