# PlanificaIA: planificador de estudio multi-agente (Hackathon 7, equipo NAN)

Sistema multi-agente hecho con **CrewAI + Ollama** (compatible con **Groq**) que recomienda un plan semanal de estudio a partir de los cursos, la disponibilidad, la dificultad y las prioridades del estudiante.

## Entregables

| # | Entregable | Archivo |
|---|---|---|
| 1 | Jupyter Notebook funcional y ejecutado | `Hack7_Planificador_Estudio_NAN.ipynb` |
| 2 | Diagrama del flujo de agentes | `diagrama_flujo.png` (también en Mermaid dentro del notebook) |
| 3 | Evidencias de ejecución | `salidas/` (salida de cada agente, traza y gráficos), `evidencia_v1_diseno_inicial/` (primera versión del diseño) y `Hack7_Planificador_Estudio_NAN.html` |
| 4 | Presentación ejecutiva (8 min) | `Presentacion_PlanificaIA_NAN.pptx` |

## Agentes
1. **Analista del estudiante**: resume contexto, prioridades, riesgos y datos faltantes.
2. **Planificador**: genera 3 planes, uno por enfoque (urgencia, balance, riesgo), asignando cursos a franjas libres.
3. **Auditor de factibilidad** ⭐ *(agente adicional)*: combina un verificador determinista en Python con la interpretación del LLM, y activa un bucle de corrección.
4. **Evaluador**: asigna puntajes de 1 a 5 según 5 criterios ponderados (el puntaje ponderado se calcula en Python).
5. **Decisor**: integra todo y recomienda el plan final, con sus riesgos y lo que debe revisar una persona.

## Cómo ejecutar

```bash
# 1. Entorno
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2a. Opción local (por defecto): Ollama
ollama pull qwen2.5:7b          # el notebook crea la variante con contexto de 16K (Modelfile)

# 2b. Opción nube: Groq
cp .env.example .env            # PROVEEDOR_LLM=groq y GROQ_API_KEY=...

# 3. Ejecutar
jupyter lab Hack7_Planificador_Estudio_NAN.ipynb   # o: jupyter nbconvert --to notebook --execute --inplace ...
```

Con Ollama en una laptop M4 Pro (24 GB), la ejecución completa toma entre 10 y 15 minutos.

## Estructura
```
Hack7/
├── Hack7_Planificador_Estudio_NAN.ipynb   # notebook entregable (ejecutado)
├── diagrama_flujo.png
├── Modelfile                              # qwen2.5:7b + num_ctx 16384
├── salidas/                               # evidencia de la ejecución final
├── evidencia_v1_diseno_inicial/           # ejecución de la primera versión (para la comparación de 6.8)
├── requirements.txt / .env.example
└── _build/                                # generador del notebook y de la presentación
```
