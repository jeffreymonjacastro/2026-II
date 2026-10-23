# Hackathon 8: priorización de tareas con Qwen3.5-4B

La entrega final incluye un **factorial 3 × 2 por duplicado**, una superficie de respuesta, comprobación real fuera del dominio y comparación del LoRA seleccionado con DoRA, rsLoRA y LoRA+ de Hugging Face PEFT. Todos los datos son sintéticos y están identificados como tales.

## Resultado y archivos principales

- `Informe_Hackathon8.pdf`: informe final de seis páginas.
- `Exposicion_Hackathon8.pptx`: presentación ejecutiva de trece diapositivas.
- `03_diseno_factorial.ipynb`: notebook ejecutado con el diseño, las doce condiciones, regresión, mejora marginal y selección final.
- `01_entrenamiento.ipynb` y `02_evaluacion.ipynb`: etapa inicial ejecutada, con aviso y resumen actualizado al final.
- `factorial/DISEÑO.md`: decisiones metodológicas y reglas fijadas antes de observar las nuevas ejecuciones.

La configuración recomendada para LoRA es **r=1, alpha=32**, seleccionada únicamente por validación según la tolerancia práctica de 0.02 de F1. Su F1 medio de validación es 0.828. En los 128 casos nuevos, el F1 es 0.794 para base, 0.835 para LoRA, 0.838 para DoRA, 0.855 para rsLoRA y 0.709 para LoRA+. Los resultados completos y sus límites están en el informe y `factorial/conclusion.md`.

## Diseño experimental ejecutado

Se cruzan r=2,8,32 con alpha=8,32. Las seis combinaciones se entrenan en una primera ronda con semilla 42 y se repiten con 43: doce entrenamientos independientes. Las esquinas se ejecutan primero; el orden dentro de las fases está fijado en `factorial/plan.json`.

Se ajusta una superficie cuadrática en log2(r), con alpha, interacción y bloque de semilla. Si su candidato requiere un rango nuevo, se entrena por duplicado. Se mide también r=64 fuera del dominio original [2,32]. Las comprobaciones se incorporan a la selección usando solo validación. LoRA, DoRA, rsLoRA y LoRA+ se comparan con el mismo rango, alpha, presupuesto y ambas semillas. Total de nuevos entrenamientos: **24**, además del modelo base sin adaptación.

rsLoRA cambia la escala a alpha/sqrt(r). LoRA+ usa el optimizador oficial de PEFT con AdamW, tasa A=0.0002 y B=0.0032, razón 16. El calendario conserva la razón en cada paso, comprobada en los historiales. Son comparaciones con una configuración compartida, no óptimos independientes de cada método. La mayor media de validación define la preferencia entre tecnologías. Los tres intervalos frente a LoRA son exploratorios y no se ajustan por multiplicidad.

Si la selección final es r=1, las escalas de LoRA y rsLoRA coinciden. En ese caso se agrega un control duplicado de rsLoRA con r=32 frente al LoRA del factorial. El control no cambia la configuración compartida y se informa por separado.

Se fijan las semillas sin forzar el uso exclusivo de kernels deterministas de CUDA. No se promete identidad a nivel de bits; las diferencias entre LoRA y rsLoRA en r=1 no se atribuyen al escalamiento. Consulte la [documentación de PyTorch](https://docs.pytorch.org/docs/stable/notes/randomness.html).

Se registran F1, errores de formato y decisión, recuperación crítica, casos difíciles, parámetros, VRAM, entrenamiento, latencia y costo hipotético. La inferencia usa lote uno y adaptadores sin fusionar. La ausencia de significación no se presenta como prueba de saturación. Se contrastan diferencias con el margen práctico 0.02 e intervalos por remuestreo pareado.

## Datos y confirmación

Los 384 ejemplos iniciales se conservan: 256 de entrenamiento, 48 de validación, 64 de prueba y 16 difíciles. Se agregan 128 casos nuevos, con familias separadas, para la comprobación final después de seleccionar hiperparámetros. Cada partición mantiene cuatro clases equilibradas. Esto representa el ejercicio sintético, no frecuencias reales de una universidad.

La prueba inicial ya había sido analizada antes del factorial. Por eso las conclusiones finales muestran también la nueva confirmación, que no se usa para decidir hiperparámetros. La transferencia a mensajes reales requiere una evaluación humana independiente.

## Ejecución en Khipu

El factorial corresponde al job **58225**, nodo **g002**, **NVIDIA RTX A6000**, 7 shards, 4 CPU y 32 GiB de RAM. La extensión rsLoRA/LoRA+ se ejecuta después, con el mismo equipo y recursos. Los tiempos antiguos del nodo ds001 se conservan como antecedentes y no se mezclan con el factorial. Consulte `factorial/results/slurm_accounting.csv` y `additional_peft_hardware.json` para los jobs y estados finales.

En el nodo de acceso, dentro del directorio del proyecto:

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python setup.py
./venv/bin/python src/check_data.py
FACTORIAL_JOB_ID=$(sbatch --parsable run_factorial.slurm)
sbatch --dependency=afterok:$FACTORIAL_JOB_ID run_peft.slurm
```

El entrenamiento y la inferencia se ejecutan únicamente en la asignación GPU de Slurm. `setup.py` descarga la revisión fijada y copia las cabeceras C del nodo de acceso cuando están disponibles; `run_factorial.slurm` configura CPATH para los kernels. Se excluyen del ZIP los pesos base y el entorno. Se incluyen las versiones de cada etapa y los adaptadores seleccionados.

Si se cambia la ubicación del proyecto, se deben ajustar las rutas absolutas de salida y el directorio de trabajo en ambos archivos Slurm. `src/predict.py --method best` usa el método con mayor media de validación y la configuración seleccionada; admite elegir método y semilla de forma explícita. Debe ejecutarse dentro de una asignación GPU.

Para repetir desde cero, trabaje en una copia con `factorial/results/` vacío y sin los adaptadores `fx_*` de la ejecución anterior. El runner puede reutilizar condiciones completas guardadas; no mezcle planes o datos diferentes. Para reconstruir el análisis desde las predicciones:

```bash
python -m pip install -r requirements-analysis.txt
python src/factorial_analysis.py
```

## Evidencia y organización

- `src/factorial_runner.py`: ejecución, réplicas, propuesta y confirmaciones.
- `src/compare_peft.py`: extensión rsLoRA y LoRA+, dos semillas por método y confirmación.
- `src/factorial_analysis.py`: tablas, regresión, remuestreo y figuras a partir de respuestas reales.
- `factorial/results/experiments.csv`: una fila por entrenamiento.
- `results/`: registros de la etapa inicial con r=8 y alpha=16. `factorial/results/` contiene la ampliación y las comparaciones finales.
- `factorial/results/cells.csv`: medias, variación y frontera de Pareto entre configuraciones de LoRA; el indicador queda vacío para los otros métodos.
- `factorial/results/*_predictions.jsonl`: todas las respuestas crudas.
- `factorial/results/theoretical_optimum.json` y `optimum_check.json`: predicción y contraste real.
- `factorial/results/selection.json`: elección por validación.
- `factorial/results/hardware.json`: asignación y hashes de los datos.
- `adapters/`: adaptadores iniciales, los nuevos LoRA, DoRA, rsLoRA y LoRA+ seleccionados y el control de rsLoRA con r=32. Los demás pesos intermedios se conservan en Khipu; sus métricas y predicciones se incluyen en el ZIP.
- `originales/`: material proporcionado por el docente, sin cambios.

El costo simulado usa USD 1/h del recurso GPU asignado e incluye solo entrenamiento. No es una tarifa ni un pago a Khipu. La GPU compartida y las dos semillas limitan la precisión de las comparaciones.

## Fuentes

Material del docente en `originales/`, con atribución del generador a Javier Ideami. Video sobre factoriales y transcripciones facilitadas: https://www.youtube.com/watch?v=EYKfpKXErtQ. Qwen: https://huggingface.co/Qwen/Qwen3.5-4B. PEFT: https://huggingface.co/docs/peft/en/package_reference/lora. DoRA: https://arxiv.org/abs/2402.09353. rsLoRA: https://arxiv.org/abs/2312.03732. LoRA+: https://arxiv.org/abs/2402.12354. Khipu: https://docs.khipu.utec.edu.pe/.

Se agradece al clúster HPC Khipu de UTEC por los recursos computacionales.
