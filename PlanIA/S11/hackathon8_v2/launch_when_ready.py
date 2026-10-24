"""Espera la descarga sin reservar GPU y después envía el trabajo a Slurm."""
import json
import subprocess
import time
from pathlib import Path
from huggingface_hub import model_info
root=Path(__file__).resolve().parent
# Verificar que cada shard exista con su tamaño definitivo antes de enviar.
info=model_info('Qwen/Qwen3.5-4B',files_metadata=True)
weights=[s for s in info.siblings if s.rfilename.endswith('.safetensors')]
while not all((root/'model'/s.rfilename).exists() and (root/'model'/s.rfilename).stat().st_size==s.size for s in weights):
    time.sleep(15)
(root/'model_revision.json').write_text(json.dumps({'model':'Qwen/Qwen3.5-4B','revision':info.sha},indent=2))
(root/'SETUP_READY').write_text('ready')
r=subprocess.run(['sbatch','--parsable',str(root/'run.slurm')],capture_output=True,text=True,check=True)
(root/'job_id.txt').write_text(r.stdout.strip())
print('SUBMITTED',r.stdout,flush=True)
