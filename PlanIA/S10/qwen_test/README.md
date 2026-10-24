# qwen_test

Este proyecto usa su propio entorno `.venv` y `uv.lock`, independiente del
proyecto padre `2026-II`, porque requieren versiones diferentes de PyTorch.
La versión seleccionada de Python es 3.12.

Desde esta carpeta, instala las dependencias y activa el entorno:

```bash
uv sync --locked
source .venv/bin/activate
```

Para actualizar las dependencias a partir de `requirements.txt`:

```bash
uv add -r requirements.txt
```

En Jupyter selecciona el intérprete de `qwen_test/.venv/bin/python`.
Para registrar un kernel con un nombre reconocible:

```bash
.venv/bin/python -m ipykernel install --user --name qwen-test --display-name 'Qwen test (Python 3.12)'
```

Seleccionar el intérprete desde **Python Environments** permite usar las
dependencias, pero no reserva una GPU ni inicia un job Slurm.

## Ejecutar automáticamente sin servidor ni kernel Jupyter

`qwen_test.py` contiene la consulta al modelo y sus métricas en Python estándar.
Para ejecutarlo, envía el script batch desde esta carpeta:

```bash
sbatch run_qwen.sbatch
```

Slurm devuelve un ID de job y la terminal queda libre. Puedes desconectarte;
la ejecución comenzará cuando haya recursos disponibles. Consulta el estado y
la salida sustituyendo `JOB_ID` por el ID recibido:

```bash
squeue -j JOB_ID
tail -f slurm-qwen-JOB_ID.out
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
```

`Ctrl+C` detiene el seguimiento de `tail`, no el job. Si necesitas cancelar
esta ejecución, usa `scancel JOB_ID` con su ID específico.

`run_qwen.sbatch` solicita los recursos y ejecuta directamente
`.venv/bin/python -u qwen_test.py`. La respuesta, las métricas y los posibles
errores quedan en `slurm-qwen-JOB_ID.out`. Esta modalidad no inicia servidor ni
kernel Jupyter y no convierte el notebook durante el job.

Para cambiar la consulta, edita `prompt` o `max_new_tokens` en `qwen_test.py`.
Las ediciones del notebook no se copian automáticamente al archivo Python.

Si ya estás dentro de una asignación Slurm con GPU, puedes ejecutar directamente:

```bash
.venv/bin/python -u qwen_test.py
```

En ese caso no necesitas enviar otro `sbatch`. El archivo `.py` contiene el
código del modelo; el archivo `.sbatch` configura la reserva y lo lanza desde el
nodo de acceso. Convertir a Python no elimina la necesidad de reservar la GPU
ni la espera en la cola.

La reserva pide 4 CPUs, 16 GiB de RAM, 4 shards de RTX A6000 y 15 minutos. No fija
un nodo: según Slurm, el 2026-10-23 puede usar `g002` o `ds001` en la partición
`gpu`, en lugar de esperar exclusivamente por una instancia de `ag001`. Los
shards comparten la GPU y no garantizan una cantidad aislada de VRAM.

`sbatch` sigue usando la cola de Slurm; evita mantener una terminal interactiva
abierta durante la espera. La disponibilidad, la prioridad y los límites de
cuenta determinan cuándo empieza. Ejecutar el archivo Python directamente en
el nodo de acceso no proporciona una GPU y no es el modo de ejecución del
proyecto en Khipu.

## Ejecutar desde VS Code con GPU

Si aparece `RuntimeError: En Khipu ejecuta este notebook dentro de un job Slurm
con GPU.`, el kernel no tiene una asignación Slurm con GPU utilizable. El
entorno `.venv` contiene las dependencias; Slurm asigna los recursos donde se
ejecutan. Sigue estos pasos para conectar VS Code a Jupyter en el nodo de cómputo.

### 1. Solicitar una GPU

Desde una terminal de VS Code conectada por SSH al nodo de acceso de Khipu,
entra a esta carpeta y solicita una sesión de 30 minutos:

```bash
cd /home/jeffrey.monja/Github/2026-II/PlanIA/S10/qwen_test
srun --account=pregrado --partition=gpu --nodelist=ag001 \
  --gres=gpu:a100_2g.10gb:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=16G --time=00:30:00 --pty bash
```

Espera a que Slurm asigne los recursos y abra la terminal del nodo de cómputo.
La solicitud puede permanecer en cola hasta que haya recursos disponibles.

### 2. Iniciar Jupyter en el nodo de cómputo

Dentro de esa terminal inicia Jupyter con el entorno del proyecto:

```bash
cd /home/jeffrey.monja/Github/2026-II/PlanIA/S10/qwen_test
hostname
.venv/bin/python -m jupyterlab --no-browser --ip=0.0.0.0 \
  --port=8888 --ServerApp.port_retries=0 --ServerApp.allow_remote_access=True
```

Mantén esa terminal abierta. Jupyter mostrará una URL con `?token=...`.

### 3. Conectar el notebook desde VS Code

1. Abre `qwen_test.ipynb`.
2. Selecciona **Select Kernel → Select Another Kernel → Jupyter Servers**.
3. Agrega la URL que muestra Jupyter. Desde VS Code conectado por SSH a Khipu,
   reemplaza `localhost` por `ag001`, conservando el puerto y el token:

   ```text
   http://ag001:8888/lab?token=EL_TOKEN_QUE_APAREZCA
   ```

4. Selecciona **Qwen test (Python 3.12)**.
5. Ejecuta las celdas de arriba hacia abajo, o usa **Run All**.

La URL con `localhost` apunta al nodo donde se ejecuta Jupyter, no al nodo de
acceso. Abrir una terminal con `srun` no cambia por sí solo el kernel de un
notebook abierto: debes conectarlo al servidor Jupyter de esa asignación.

### 4. Cerrar la sesión y liberar los recursos

Al terminar, detén Jupyter con `Ctrl+C` y confirma su cierre; después ejecuta
`exit` para liberar la sesión Slurm. La sesión también finaliza al alcanzar los
30 minutos. Si el puerto 8888 está ocupado, elige otro puerto tanto en el
comando como en la URL.

Los recursos se verificaron el 2026-10-23; revisa Slurm antes de cambiar el
tamaño de la reserva. Sigue `AGENTS.md` para los límites de cuenta. El checkpoint
se reutiliza desde `../Lora Scripts/models/`; no se descarga al ejecutar.

Referencias: [jobs interactivos de Khipu](https://docs.khipu.utec.edu.pe/guia-de-usuario/enviar-jobs/jobs-interactivos/)
y [servidores Jupyter en VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks#_connect-to-a-remote-jupyter-server).
