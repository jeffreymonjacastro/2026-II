# Semana 9: CrewAI en Kaggle

Notebook basado en `../Sem9_Course_MultiAgent_CrewAI.ipynb`. Conserva la API KEY de ejemplo, el modelo `groq/openai/gpt-oss-20b`, el tópico `Aborto` y la consulta de soporte de DeepLearningAI.

## Versiones

- `v1/input/`: primera adaptación. Blog completó tres tareas; soporte falló porque Groq rechazó el esquema vacío de `ScrapeWebsiteTool` configurado con URL fija. Evidencia en `v1/outputs/run_summary.json`.
- `v2/input/`: corrección del esquema del tool. Blog completó tres tareas; soporte excedió el límite de Groq: 14.964 tokens solicitados frente a 8.000 TPM. Evidencia en `v2/outputs/run_summary.json`.
- `v3/input/`: completó las cinco tareas, pero la respuesta final de soporte quedó truncada con el límite de 1.500 tokens.
- `v4/input/`: contexto de 3.500 caracteres por página, hasta 3.500 tokens de salida y `reasoning_effort="low"`; respuesta de soporte limitada a 600 palabras para evitar truncamiento. Esta es la versión para usar. [Parámetro de razonamiento documentado por Groq](https://console.groq.com/docs/reasoning).

Cada versión incluye notebook y `kernel-metadata.json`; sus archivos descargados se guardan en `outputs/`.

## Ejecutar

Desde `v4/input/`, con Kaggle CLI autenticada:

```powershell
kaggle kernels push -p . --accelerator NvidiaTeslaT4
kaggle kernels status jeffreyamc/sem9-multiagent-crewai-kaggle
$env:PYTHONUTF8 = '1'
kaggle kernels output jeffreyamc/sem9-multiagent-crewai-kaggle -p ..\outputs --force
```

Notebook remoto privado: [Sem9 MultiAgent CrewAI Kaggle](https://www.kaggle.com/code/jeffreyamc/sem9-multiagent-crewai-kaggle).

Internet y T4 están habilitados. Las inferencias usan Groq; la GPU no interviene en ellas. Instalación fijada a CrewAI y tools `1.15.27`, LiteLLM `1.89.2` y Groq `0.37.1`. No hace falta reiniciar el kernel.

## Resultados

Versión 4 ejecutada en Kaggle con estado remoto `COMPLETE` y `run_summary.json` con `status: complete`: tres tareas de blog y dos de soporte. Terminó el 13 de octubre de 2026 a las 23:11:48, hora de Lima. Se descargaron siete archivos Markdown, el resumen JSON y el log. El log confirma consultas reales a las dos páginas de documentación; la respuesta final de soporte ya termina completa. Los ejemplos de código generados por los agentes no se ejecutaron ni validaron por separado.

- `blog_plan.md`, `blog_draft.md`, `blog_edited.md` y `blog.md`: planificación, borrador y artículo final.
- `support_draft.md`, `support_reviewed.md` y `support.md`: respuesta inicial y revisada.
- `run_summary.json`: versiones, estado de cada flujo, número de tareas y errores sin API KEY.

`COMPLETE` de Kaggle indica que terminó el notebook. El éxito de los agentes requiere además `status: complete` en `run_summary.json`, con tres tareas de blog y dos de soporte.

El blog es contenido generado sin búsqueda web; sus afirmaciones requieren verificación antes de reutilizarse. Soporte utiliza scraping de documentación oficial de crews y memoria, con recorte explícito del contexto para respetar la cuota de Groq. La memoria permanece desactivada, como en el original. No se envían mensajes al cliente.
