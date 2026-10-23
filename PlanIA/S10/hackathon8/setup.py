"""Preparación en el nodo de acceso: dependencias y pesos; no realiza cómputo GPU."""
from pathlib import Path
from huggingface_hub import snapshot_download, model_info
import json
import shutil
root=Path(__file__).resolve().parent
if Path('/usr/include/stdlib.h').exists() and not (root/'compiler_include').exists():
    # Algunos nodos GPU de Khipu no incluyen cabeceras C para compilar Triton.
    shutil.copytree('/usr/include',root/'compiler_include',symlinks=True)
revision_file=root/'model_revision.json'
revision=json.loads(revision_file.read_text())['revision'] if revision_file.exists() else 'main'
info=model_info('Qwen/Qwen3.5-4B',revision=revision)
(root/'model_revision.json').write_text(json.dumps({'model':'Qwen/Qwen3.5-4B','revision':info.sha},indent=2))
snapshot_download('Qwen/Qwen3.5-4B',revision=info.sha,local_dir=root/'model',
                  ignore_patterns=['*.md','*.png','*.jpg'])
(root/'SETUP_READY').write_text('ready')
print('SETUP_READY')
