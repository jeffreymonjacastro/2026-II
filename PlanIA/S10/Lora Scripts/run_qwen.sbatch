#!/bin/bash
#SBATCH --job-name=qwen35-prioridades
#SBATCH --account=pregrado
#SBATCH --partition=gpu
#SBATCH --gres=shard:rtxa6000:7
#SBATCH --nodelist=g002
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export HF_HOME="$PWD/models/hf-home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export QWEN_LOCAL_ONLY=1
export QWEN_MODE="${1:-deadline}"
export QWEN_RUN_ID="${SLURM_JOB_ID}_${QWEN_MODE}"
export IPYTHONDIR="$PWD/.cache/ipython"
export JUPYTER_DATA_DIR="$PWD/.cache/jupyter"
export JUPYTER_RUNTIME_DIR="$PWD/.cache/jupyter-runtime"
mkdir -p "$MPLCONFIGDIR" "$IPYTHONDIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR"

if [[ "$QWEN_MODE" != full && "$QWEN_MODE" != smoke && "$QWEN_MODE" != deadline ]]; then
    echo 'Expected: sbatch run_qwen.sbatch [full|smoke|deadline]' >&2
    exit 2
fi
if [[ ! -f models/model_manifest.json ]]; then
    echo 'Download the model first using the download cell in the notebook.' >&2
    exit 2
fi
# Run the project Python as the Jupyter kernel, independent of the login kernel.
.venv/bin/python -m ipykernel install --prefix "$PWD/.cache/kernel" --name qwen-project
export JUPYTER_PATH="$PWD/.cache/kernel/share/jupyter"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
# Keep cell outputs and traceback even if execution fails; experiment code stays in the notebook.
# The batch step already owns the allocation. Running directly also supports
# shard allocations, whose GRES may not propagate correctly into an srun step.
.venv/bin/python -u - <<'PY'
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

notebook = nbformat.read('qwen35_4b_priorizacion.ipynb', as_version=4)
def report_cell(cell, cell_index, **kwargs):
    print('Completed cell', cell_index, cell.metadata.get('tags', []), flush=True)

client = NotebookClient(notebook, timeout=13200, kernel_name='qwen-project',
                        resources={'metadata': {'path': str(Path.cwd())}},
                        on_cell_executed=report_cell)
try:
    client.execute()
except BaseException:
    destination = Path('outputs') / (os.environ['QWEN_RUN_ID'] + '_failed.ipynb')
    nbformat.write(notebook, destination)
    print('Execution failed; partial notebook:', destination, flush=True)
    raise
else:
    destination = Path('outputs') / (os.environ['QWEN_RUN_ID'] + '_executed.ipynb')
    nbformat.write(notebook, destination)
    print('Execution completed:', destination, flush=True)
PY
