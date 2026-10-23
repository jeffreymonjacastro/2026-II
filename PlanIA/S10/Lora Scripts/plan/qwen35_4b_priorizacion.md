# Qwen3.5-4B: priorización de tareas, LoRA y DoRA

## Objetivo y alcance acordado

Crear un único notebook nuevo basado en main.ipynb, sin modificar los archivos
del docente ni las ediciones existentes. Explicaciones y ejemplos en español.
Comparar el checkpoint original Qwen/Qwen3.5-4B con un entrenamiento LoRA y otro
DoRA independiente mediante Hugging Face PEFT. Todos los datos, funciones,
experimentos, evaluaciones y gráficos viven en el notebook.

Se omiten por acuerdo el diseño factorial, réplicas con varias semillas, 20% de
datos escritos por alumnos, regresión, análisis monetario y presentación. No se
afirma cumplir toda la rúbrica original ni demostrar superioridad general.

## Implementación

1. Revisar requerements.txt y conservarlo; crear requirements.txt compatible con
   Linux y CUDA, con Matplotlib y scikit-learn. Congelar versiones efectivas.
2. Descargar el checkpoint oficial completo antes del cómputo GPU, guardar la
   revisión resuelta y reutilizarlo localmente en modo offline.
3. Crear qwen35_4b_priorizacion.ipynb: objetivo, preparación, configuración,
   política, datos, baseline, LoRA, DoRA, resultados, gráficos y conclusiones.
4. Clasificar tareas TI con JSON {priority, reason} y cuatro clases. P1: incidente
   activo de seguridad/pérdida de datos o caída crítica sin alternativa. P2:
   trabajo bloqueado sin alternativa o plazo con consecuencias <=24 h. P3:
   degradación no crítica, alternativa disponible o plazo con consecuencias
   <=7 días. P4: mantenimiento/mejoras opcionales sin impacto inmediato.
5. Generar 600 ejemplos sintéticos únicos, balanceados: 400 entrenamiento, 100
   validación, 100 prueba. Separar familias de escenarios/plantillas, fijar
   semilla, verificar etiquetas y ausencia de duplicados y guardar huellas.
6. Cargar la clase nativa de generación condicional; congelar visión. Usar el
   chat template oficial y enable_thinking=False. Baseline con generación greedy,
   máximo 128 tokens, batch 1, calentamiento fuera del test.
7. Cuantizar consistentemente en NF4 4-bit con doble cuantización. Preparar cada
   método desde una carga nueva. Adaptar solo MLP del modelo de lenguaje
   (gate_proj/up_proj/down_proj), comprobar nombres y parámetros entrenables.
8. Ajustes compartidos: r=16, alpha=32, dropout=0, lr=2e-4, 3 épocas, batch 1,
   acumulación 16, máximo 512 tokens, warmup_ratio=0.03, semilla 42. Gradient
   checkpointing y pérdida únicamente sobre la respuesta. Validación por época,
   selección por menor pérdida de validación. DoRA: use_dora=True. Evaluación sin
   fusionar adaptadores. Mantener idénticos datos, hardware y presupuesto.
9. Guardar outputs/<run_id>/: configuración, entorno, paquetes, dataset JSONL,
   predicciones CSV, métricas CSV/JSON, historial, adaptadores y PNG Matplotlib.
   Medir exactitud, macro-F1 de cuatro clases, recall por clase, JSON/esquema,
   latencia, tiempo de entrenamiento, parámetros y picos de VRAM. Respuestas
   inválidas cuentan como errores; conservar columna inválida en confusión.
10. Figuras: distribución de clases, comparación de calidad, confusiones,
    curvas de pérdida y recursos. Conclusiones basadas en resultados reales.

## Khipu y validación

Seguir AGENTS.md: preparación/transferencias en login; inferencia y entrenamiento
solo con Slurm. Revisar límites antes de enviar. Solicitud objetivo: gpu,
rtxa6000:1, 4 CPUs, 48 GiB RAM, 4 horas. Smoke de 30 minutos, datos reducidos y
dos pasos por método. El wrapper del scheduler solo ejecuta el notebook.

Validar formato, política, separación, máscaras de etiquetas, scoring de inválidos,
gradientes finitos/no nulos, actualización, guardado/recarga de ambos adaptadores.
Ejecutar de principio a fin e inspeccionar resultados y figuras. Registrar
restricciones reales de entorno o cola sin inventar resultados ni cambiar solo
un método. Aceptación: notebook ejecutado, tres condiciones medidas, adaptadores
recargables, CSV/JSON reconciliables y gráficos legibles. Mejorar métricas no es
un requisito de aceptación.

Referencias: https://huggingface.co/Qwen/Qwen3.5-4B ;
https://huggingface.co/docs/peft/package_reference/lora_variant_dora ; AGENTS.md.

## Ajuste por entrega en cuatro horas (2026-10-23)

La cola dedicada no cumple el plazo. Se cancelaron únicamente los jobs propios
58316/58317. El smoke 58318 se ejecuta en g002 con siete shards RTX A6000,
cuatro CPUs y 48 GiB RAM; el nodo informó ~16 GiB VRAM libres. No se alteró
ningún proceso o job de otros usuarios.

Se incorpora el modo `deadline`: 200 ejemplos de entrenamiento, 40 validación,
40 prueba y una época por adaptador. Se mantienen checkpoint, r=16, alpha=32,
semilla, política y condiciones comparables. La comparación sigue incluyendo
original, LoRA y DoRA; el tamaño reducido y la GPU compartida se declaran como
limitaciones. El modo full permanece disponible para trabajo posterior.
