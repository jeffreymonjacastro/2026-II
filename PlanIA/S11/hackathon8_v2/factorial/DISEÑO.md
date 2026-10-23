# Diseño factorial y criterio de decisión

La pregunta es si aumentar el rango de LoRA sigue mejorando la priorización de tareas y si ese efecto depende de alpha. Se estudian conjuntamente desempeño, memoria y tiempo, manteniendo el presupuesto de aprendizaje.

## Factores, niveles y réplicas

| Factor | Niveles |
|---|---|
| Rango r | 2, 8 y 32 |
| Alpha | 8 y 32 |
| Réplica | Semilla 42 y semilla 43 |

El cruce completo produce seis combinaciones y doce entrenamientos. La semilla es un bloque de réplica, no un tercer hiperparámetro que se optimice. La primera ronda utiliza 42 en todas sus condiciones y la segunda 43. El orden se fija con aleatorización reproducible dentro de las fases; las cuatro esquinas se prueban primero y luego se completa el rango intermedio. La secuencia exacta se conserva en `plan.json`.

Se mantienen Qwen3.5-4B, revisión de pesos, NF4, datos y particiones, prompt, proyecciones adaptadas, dropout 0.05, tasa máxima 0.0002, 32 actualizaciones y lote efectivo 8. Cada adaptador parte del modelo original. En LoRA estándar la escala es alpha/r: aumentar r con alpha fijo también cambia esa escala. Por ello se estudia la interacción y no se atribuye todo cambio exclusivamente a capacidad.

## Hardware y medición

El factorial, sus comprobaciones y DoRA se ejecutan en una asignación de Slurm: g002, NVIDIA RTX A6000, 7 shards, 4 CPU y 32 GiB de RAM. La comparación adicional con rsLoRA y LoRA+ usa una segunda asignación con esos mismos componentes. La GPU se comparte, de modo que los tiempos pueden variar por carga externa entre condiciones y asignaciones. Los tiempos de la etapa inicial en ds001 se conservan como antecedentes y no se combinan con los del factorial.

Antes de medir cada entrenamiento se realiza una pasada forward/backward sin actualizar pesos. Se descartan sus gradientes y se restaura la semilla. Este calentamiento prepara kernels y se registra por separado. La VRAM se mide con el máximo asignado por PyTorch durante el entrenamiento, después de reiniciar sus contadores. La latencia usa lote uno, decodificación sin muestreo, sincronización CUDA y calentamiento excluido. Los adaptadores no se fusionan.

Cada condición registra F1 macro de validación y prueba, formato JSON, coincidencia de campos, recuperación crítica, casos difíciles, parámetros entrenables, VRAM de entrenamiento e inferencia, tiempo, latencia y costo hipotético de entrenamiento a USD 1/h. Este precio es una hipótesis comparativa, no una tarifa de Khipu.

## Análisis y óptimo práctico

Se ajusta una superficie cuadrática en log2(r), con alpha como factor de dos niveles, interacciones y bloque de semilla. Sus seis medias de celda y el efecto de réplica se estiman con doce observaciones. El análisis informa los grados de libertad y trata las pruebas F como exploratorias. Dos niveles de alpha no permiten estimar una curvatura propia de ese factor.

La regla de selección, fijada antes de las nuevas ejecuciones, considera equivalentes a efectos de decisión las configuraciones cuya media de F1 de validación esté a menos de 0.02 del máximo observado. Entre ellas se elige el menor rango, luego el menor tiempo medio de entrenamiento. Ese margen es una preferencia práctica, no una afirmación automática de equivalencia estadística. La frontera de Pareto complementa la regla con F1, memoria y tiempo.

Se remuestrean conjuntamente casos y pares de semillas para estimar diferencias marginales entre rangos adyacentes observados, incluyendo los niveles nuevos de comprobación. Una p no significativa no demuestra saturación. Solo si el límite superior unilateral del 95 % queda bajo 0.02 se considera que los datos acotan una mejora relevante en ese tramo, dentro de las semillas y casos observados. Se informa el primer tramo compatible con ese criterio a partir de la configuración seleccionada; los demás intervalos se conservan y no se interpreta una garantía global. Con dos réplicas no se caracteriza toda la variación posible del entrenamiento.

## Comprobación del modelo y fuera del dominio

La superficie se ajusta únicamente con las doce validaciones del factorial. Se busca su máximo previsto en rangos enteros de 1 a 64 y los dos alpha estudiados. Si propone un rango nuevo, se entrena realmente por duplicado. Además se evalúa siempre r=64 con el alpha de la configuración inicialmente seleccionada. Así se contrasta una extrapolación con mediciones y se explora la frontera fuera del intervalo original [2,32]. No se supone que una predicción fuera del dominio sea válida por sí sola.

La selección se actualiza usando las validaciones de todas las configuraciones medidas. El test y los casos difíciles no deciden hiperparámetros. Como la prueba inicial ya había sido analizada, se incorpora una comprobación adicional de 128 casos sintéticos nuevos, con familias separadas. Este conjunto se abre después de fijar la selección y se usa para base, LoRA, DoRA, rsLoRA y LoRA+.

## Comparación con DoRA, rsLoRA y LoRA+

El LoRA seleccionado se contrasta con tres métodos de Hugging Face PEFT en el mismo rango, alpha, presupuesto y ambas semillas. DoRA se activa con `use_dora=True`. rsLoRA usa `use_rslora=True` y cambia la escala de alpha/r a alpha/sqrt(r). LoRA+ usa el optimizador oficial `create_loraplus_optimizer`, con AdamW y razón de tasas B/A=16. Se mantiene la tasa máxima de A en 0.0002 y B alcanza 0.0032. El mismo multiplicador del calendario coseno se aplica a cada grupo, conservando esa razón durante los 32 pasos; se comprueba y registra en cada actualización.

Las tasas distintas forman parte de LoRA+. El rango y alpha se comparten con la selección de LoRA; no se buscan óptimos independientes para cada método. Todos usan el mismo equipo. Las decisiones de esta extensión se conservan en `peft_extension_plan.json` y se fijan antes de evaluar la confirmación nueva. La mayor media de validación determina la preferencia entre métodos; los 128 casos nuevos sirven para comprobarla. Los intervalos individuales de tres comparaciones frente a LoRA son exploratorios y no se ajustan por multiplicidad.

Si la selección es r=1, las escalas de LoRA y rsLoRA coinciden. El plan agrega entonces un control con rsLoRA en r=32, alpha compartido y semillas 42 y 43, comparado con el LoRA correspondiente del factorial. Este control permite observar el mecanismo con escalas distintas. Se evalúa en validación, prueba y casos difíciles; no interviene en la elección compartida ni se selecciona usando los casos nuevos.

Se fijan las semillas de los generadores aleatorios, pero no se fuerza el uso exclusivo de kernels deterministas de CUDA. Las réplicas no prometen identidad de resultados a nivel de bits. En particular, las diferencias observadas entre LoRA y rsLoRA en r=1 no se atribuyen a la escala, que es idéntica. Referencia: https://docs.pytorch.org/docs/stable/notes/randomness.html.

La conclusión corresponde a esta tarea sintética y presupuesto; no establece un límite universal de LoRA ni una superioridad general entre tecnologías. El código exacto empleado en el factorial se conserva en `source_factorial/`, junto con la ampliación del código principal para los métodos adicionales.

## Evidencia reproducible

- `plan.json`: factores, orden y reglas fijadas antes de observar resultados.
- `results/experiments.csv`: una fila por entrenamiento real.
- `results/*_predictions.jsonl`: respuestas crudas por ejemplo.
- `results/regression.json`: ajuste y pruebas exploratorias.
- `results/marginal_gains.json`: diferencias e intervalos.
- `results/theoretical_optimum.json`: propuesta del modelo, antes de comprobarla.
- `results/selection.json`: elección final basada en validación.
- `results/hardware.json`: GPU, nodo, job y hashes de los datos.

Referencia conceptual: video «Diseños factoriales. Diseños de investigación en Psicología 8/10», https://www.youtube.com/watch?v=EYKfpKXErtQ, y transcripciones proporcionadas. Implementación de LoRA y DoRA: https://huggingface.co/docs/peft/en/package_reference/lora.
