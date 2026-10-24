# Qwen3.5-4B: prioridades por tarea en conjuntos

## Definición final confirmada por el usuario

Recibir un conjunto de tareas y asignar a cada una Critical, High, Medium o Low,
con una razón breve. Conservar IDs. El orden de salida no afecta la clasificación.
No construir horarios. Todos los experimentos permanecen en un notebook nuevo;
los originales del docente se conservan. Los intentos previos se archivan y no
se utilizan como evidencia de esta comparación.

## Política y datos

Cada tarea contiene id, descripción, deadline_hours, impact (1–5), blocked,
workaround y active_incident. Aplicar en orden: Critical ante incidente activo o
bloqueo crítico sin alternativa; High ante bloqueo sin alternativa o plazo <=24h
con impacto >=3; Medium ante alternativa, plazo <=168h o impacto >=3; Low en los
demás casos. La insistencia del texto no cambia los atributos explícitos.
Salida JSON tasks con id, priority y reason por tarea. Razones revisadas de forma
cualitativa, sin inferir que sean correctas solo por existir.

Generar datos sintéticos reproducibles y etiquetas derivadas de la política.
Equilibrar prioridades globalmente por partición, no imponer una de cada por set.
Separar servicios/redacciones y comprobar ausencia de duplicados. Deadline:
80/12/12 conjuntos de cuatro tareas, una época, 48 tareas de prueba. Full opcional:
400/100/100 y tres épocas. Datos y configuración iguales para los tres modelos.

## Modelo y ejecución para entrega en cuatro horas

Checkpoint oficial descargado y revisión fijada. Qwen original, LoRA y DoRA desde
cargas independientes. NF4 4-bit, doble cuantización, r=16, alpha=32, dropout=0,
lr=2e-4, acumulación 16, límite 768 tokens, hasta 224 generados. Adaptar 96 MLP
projections del lenguaje y congelar visión. Pérdida solo en respuestas. Validación
por época y menor pérdida de validación; DoRA con use_dora=True en PEFT.

Usar Slurm con siete shards disponibles en g002, sin alterar jobs de otros usuarios.
La infraestructura GPU y adaptadores ya pasaron un smoke previo; la nueva estructura
y scoring pasan verificaciones CPU. El run final repite las comprobaciones de
gradientes, actualización, guardado y recarga. No esperar la cola dedicada de dos días.

## Evidencia y validación

Medir exactitud y macro-F1 por tarea, recall por prioridad, conjuntos enteramente
correctos, completitud, duplicados, IDs desconocidos y JSON/esquema. Alinear por
ID; omisiones y duplicados reciben INVALID y cuentan como errores. Guardar
responses.csv por set, predictions.csv por tarea, métricas CSV/JSON, dataset JSONL,
configuración, entorno, versiones, historial y adaptadores.

Matplotlib: distribución de prioridades, comparación de calidad, matrices con
columna INVALID, pérdidas y recursos. Mostrar cada tarea, prioridad y razón.
Reconciliar CSV con métricas e inspeccionar figuras. Los tiempos en GPU compartida
y picos del proceso secuencial no son benchmarks aislados. No inventar mejoras.

Omitidos: factorial, réplicas, 20% manual, regresión, costos monetarios y PPT.
Referencias: AGENTS.md; https://huggingface.co/Qwen/Qwen3.5-4B ;
https://huggingface.co/docs/peft/package_reference/lora_variant_dora .
