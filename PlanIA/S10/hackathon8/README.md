# Hackathon 8: priorización de tareas con Qwen3.5-4B

El proyecto compara el modelo original, LoRA y DoRA para asignar cuatro prioridades a tareas universitarias y de soporte de TI. La salida es un JSON con prioridad, escalamiento, bloqueo y criterio decisivo. Los datos son totalmente sintéticos.

## Entrega y resultado

- `Informe_Hackathon8.pdf`: informe de cuatro páginas.
- `01_entrenamiento.ipynb` y `02_evaluacion.ipynb`: notebooks ejecutados con los registros reales de Khipu.
- `Exposicion_Hackathon8.pptx`: presentación de nueve diapositivas con tablas y gráfica editables.

La ejecución final corresponde al job `58224`: estado `COMPLETED`, salida `0:0` y duración de 33 min 55 s. En los 64 casos de prueba, el F1 macro fue 0.760 para base, 0.876 para LoRA y DoRA, y 0.830 para la réplica de LoRA. LoRA se recomienda para este prototipo por el empate de desempeño con menor costo computacional. Se conservaron dos errores entre los 16 casos críticos en todas las condiciones.

## Comparación solicitada con una tecnología más actual

**Se entrenó y evaluó DoRA con Hugging Face PEFT para contrastarlo con el LoRA seleccionado.** DoRA separa magnitud y dirección de los pesos; en `src/experiment.py` se activa mediante `LoraConfig(..., use_dora=True)`. Los adaptadores obtenidos se entregan en `adapters/dora/` y `adapters/lora/`.

La comparación principal usa semilla 42 y conserva modelo, datos, prompt, rango, alpha, presupuesto y hardware asignado. LoRA se selecciona entre sus dos ejecuciones por F1 de validación (0.776 frente a 0.683 de la réplica). DoRA obtiene 0.776 en validación. En prueba, ambos alcanzan F1 macro 0.876, esquema válido del 100 % y recuperación crítica del 87.5 %. DoRA demanda 2.04 veces el tiempo de entrenamiento y 1.09 veces la memoria. Su latencia medida sin fusionar es 4.27 veces mayor.

Se recomienda LoRA para esta configuración. La comparación está desarrollada en la sección 6 del informe, las diapositivas 4 y 8, y el notebook de evaluación. Los resultados no establecen una superioridad general del método. El gráfico `figures/comparacion_peft.png` resume desempeño y consumo relativo.

## Archivos

- `src/dataset_generator.py`: crea los datos y aplica una rúbrica reproducible.
- `src/validator.py`: comprueba el JSON completo y sus tipos.
- `src/evaluation.py`: calcula desempeño, cumplimiento del esquema y latencia.
- `src/experiment.py`: carga Qwen en 4 bits, entrena los adaptadores y evalúa.
- `src/check_data.py`: verifica particiones y casos límite de la rúbrica.
- `src/predict.py`: aplica un adaptador a una nueva descripción de tarea, dentro de una asignación GPU.
- `data/`: entrenamiento, validación, prueba y casos de dificultad adicional.
- `results/`: resultados reales, predicciones individuales y registro de Slurm.
- `adapters/`: adaptadores entrenados de LoRA, DoRA y la réplica.
- `originales/`: material entregado por el docente, conservado sin cambios.

## Ejecución en Khipu

En el nodo de acceso, dentro de `/home/fabryzzio.meza/hackathon8_qwen35`:

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python setup.py
./venv/bin/python src/dataset_generator.py data
./venv/bin/python src/check_data.py
sbatch run.slurm
```

La descarga se realiza en el nodo de acceso. El entrenamiento y la inferencia se ejecutan únicamente en el nodo GPU asignado por Slurm. El archivo `model_revision.json` fija la revisión descargada; `requirements-lock.txt` conserva las versiones del entorno.

`setup.py` también copia las cabeceras C del nodo de acceso al espacio del proyecto, cuando están disponibles. Esto permite compilar los kernels de Triton en nodos de Khipu que no incluyen esas cabeceras. `run.slurm` configura `CPATH` para usarlas.

Para consultar el estado:

```bash
squeue --me
tail -f logs/experiment-JOBID.out
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

## Criterios de comparación

La configuración principal utiliza rango 8, alpha 16, semilla 42, 32 actualizaciones y lote efectivo de 8 ejemplos. Los dos adaptadores parten de cargas independientes del mismo modelo. La réplica de LoRA utiliza semilla 43. Todas las condiciones mantienen los mismos datos, prompt, precisión numérica, presupuesto de entrenamiento y hardware asignado.

La métrica principal es F1 macro sobre las cuatro prioridades. Una salida inválida cuenta como error; no se extraen fragmentos de JSON ni se reparan respuestas. También se registran coincidencia exacta de los cuatro campos, recuperación de casos críticos, error ordinal, VRAM de PyTorch y latencia individual tras calentamiento. El costo simulado supone USD 1 por hora del recurso GPU empleado: es una hipótesis de análisis, no una tarifa de Khipu.

## Alcance

Las etiquetas representan una política definida para este ejercicio. La prueba conserva el reparto equilibrado del conjunto sintético, sin afirmar que reproduzca la frecuencia de urgencias de una universidad. Las familias de tareas de prueba no aparecen en entrenamiento. La transferencia a solicitudes reales requiere una evaluación humana independiente.

El modelo de referencia del notebook original pertenece a Gemma; Qwen es una familia distinta. El registro del curso consultado incluye Qwen3.5-4B y DoRA entre las elecciones. La fila de Qwen3.5 no identifica al equipo, por lo que la inscripción del grupo corresponde confirmarla con el docente.

## Fuentes

- Material del docente incluido en `originales/`; scripts originales con atribución a Javier Ideami.
- Qwen: https://huggingface.co/Qwen/Qwen3.5-4B
- Transformers: https://huggingface.co/docs/transformers/en/model_doc/qwen3_5
- PEFT y DoRA: https://huggingface.co/docs/peft/en/package_reference/lora
- Khipu: https://docs.khipu.utec.edu.pe/

Se agradece al clúster HPC Khipu de la Universidad de Ingeniería y Tecnología por los recursos computacionales utilizados en este proyecto.
