# Hackatón 11 — ¿Confiar, verificar o escalar? · v1

**Versión más reciente:** [v2 en Kaggle — F1 de buen crédito y HACS](kaggle/v2/README.md).
Sus resultados ejecutados están en `kaggle/v2/outputs/hack11_v2_ejecutado.ipynb`.
La documentación y los resultados originales de v1 se conservan a continuación.

Dos notebooks autocontenidos, explicados en español y alineados con las seis partes
de `Sem13_Hackaton11.pdf`. El segundo comparte el protocolo y cambia los modelos.
**Completar los nombres del equipo** en la portada y en `GROUP` antes de entregar.

| Variante | Modelos | Notebook |
|---|---|---|
| Kaggle | XGBoost, LightGBM, CatBoost | `kaggle/v1/input/hack11_v1.ipynb` |
| Kaggle ejecutado | Mismos tres modelos, con salidas remotas | `kaggle/v1/outputs/hack11_v1_ejecutado.ipynb` |
| Local ejecutado | Random Forest, Logistic Regression | `local/hack11_local_v1.ipynb` |

- [Notebook privado en Kaggle](https://www.kaggle.com/code/jeffreyamc/hack11-human-ai-credit-v1).
- [Dataset privado](https://www.kaggle.com/datasets/jeffreyamc/hack11-german-credit-s13),
  versión 1: cinco archivos originales verificados por SHA-256 después de descargarlos.
- Vistas HTML: `kaggle/v1/outputs/hack11_v1.html` y `local/hack11_local_v1.html`.
- Plan y decisiones previas: `PLAN.md`. Evidencia de integridad: `validation_report.json`.

Estado final: **Kaggle COMPLETE, revisión remota 3; local complete/done**.
Ambos notebooks tienen 39 celdas, 19 de código ejecutadas y cero errores.
Las fórmulas se recalcularon independientemente desde conteos y decisiones;
se verificaron fuentes, particiones, SHA-256, nueve casos SHAP y diez figuras por
entorno. Las vistas HTML y los gráficos se inspeccionaron visualmente. La tabla
conjunta se conserva en `comparacion_v1.csv`.

## Protocolo

`german.data` contiene 1,000 solicitudes y 20 atributos (7 numéricos, 13 categóricos),
sin nulos ni filas duplicadas. No concatenamos la representación numérica alternativa.
`y=1` significa **mal crédito**; el código original 1 significa bueno y 2 significa malo.

- Particiones estratificadas fijas: 600 entrenamiento, 200 validación, 200 prueba.
  Los índices son idénticos en ambos entornos. No hay SMOTE.
- Imputación, one-hot y escalamiento ajustados sólo con entrenamiento.
- Selección por F1 binario de validación; ROC-AUC desempata. Umbral IA = 0.5.
- Entropía binaria con logaritmo natural: `H = -p ln(p) - (1-p) ln(1-p)`.
- Política: `H < A → IA`, `A ≤ H < B → Junior`, `H ≥ B → Senior`.
- Junior 70% y senior 95% de accuracy **simulada**; errores Bernoulli independientes.
  No son porcentajes de la métrica Precision ni resultados de personas reales.
- Optimización de HACS con 200 réplicas en validación, buscando mejora media de F1
  ≥ 0.01 y mejora en ≥ 80% de réplicas. Prueba usa 500 réplicas con otra semilla.
- No se reentrena después de fijar umbrales, ni se eligen modelos mirando prueba.
- Las políticas elegidas tienen `A=B`: los candidatos con junior se evaluaron, pero
  la búsqueda prefirió IA/senior. Se incluyen ambos simuladores y sus baselines.

## Fórmulas y resultados

F1 principal binario para **mal crédito**: `2 TP / (2 TP + FP + FN)`; F1 macro se
reporta adicionalmente. El PDF no determina la variante, por lo que se declara.

```
Mejora       = F1 colaborativo − F1 IA
Mejora_pp    = 100 × Mejora
Coverage     = casos IA / total
Efficiency   = 1 − casos revisados por humanos / total
HACS         = 0.6 × F1 colaborativo + 0.2 × Coverage + 0.2 × Efficiency
Coste        = FP + 5 × FN
```

Coverage y Efficiency coinciden porque cada solicitud tiene un único responsable.
HACS no incorpora mayor coste laboral del senior ni el coste crediticio asimétrico.
Los reportamos separadamente, sin modificar la fórmula solicitada.

La tabla reproduce la **réplica 0 de prueba**, fijada previamente; no se buscó una
semilla favorable. «Mejora» es una diferencia de F1, no un porcentaje relativo.

| Entorno | Modelo | F1 sólo IA | F1 IA + Human | Mejora | HACS |
|---|---|---:|---:|---:|---:|
| Kaggle | XGBoost | 0.5321 | 0.8037 | +0.2716 | 0.7302 |
| Kaggle | LightGBM | 0.5091 | 0.7523 | +0.2432 | 0.7054 |
| Kaggle | **CatBoost** | **0.5946** | **0.8850** | **+0.2904** | **0.7350** |
| Local | **Random Forest** | **0.5874** | **0.5915** | **+0.0041** | **0.7469** |
| Local | Logistic Regression | 0.6133 | 0.6486 | +0.0353 | 0.7612 |

Los modelos en negrita fueron elegidos por **validación**. Logistic Regression
obtiene mayor resultado en esta prueba, pero eso no cambia la elección local.
Las comparaciones de candidatos son breves; SHAP y defensa se concentran en el
seleccionado, siguiendo la indicación de desarrollar el mejor resultado.

**CatBoost:** 102 casos IA y 98 senior; cobertura 51%. El coste relativo baja de
153 a 53 en la réplica reportada. F1 colaborativo medio de 500 réplicas = 0.8435,
intervalo Monte Carlo central 95% [0.8036, 0.8750], HACS medio = 0.7101 y mejora
frente a IA en 100% de réplicas. El F1 0.8850 de la réplica reportada es superior
a ese intervalo central; no lo presentamos como resultado típico.

**Random Forest:** 196 casos IA y 4 senior; cobertura 98%. La mejora puntual es
pequeña. El intervalo bootstrap pareado de ΔF1 es [0.0000, 0.0141], por lo que
no se sostiene una mejora poblacional concluyente. Mejora en 83.6% de réplicas.

El intervalo Monte Carlo varía únicamente respuestas humanas sobre los mismos
200 casos. El bootstrap pareado varía solicitudes para la réplica 0, sin repetir
entrenamiento ni selección. Ninguno representa toda la incertidumbre de despliegue.
Los beneficios dependen de accuracies humanas supuestas, constantes incluso en
casos difíciles. SHAP no modifica esas respuestas ni demuestra causalmente que
una explicación mejoró al analista. No se ha establecido equidad ni validez externa.

## Contenido de los notebooks

1. Contexto, matriz de costes, esquema, estadísticas y EDA con cuatro vistas.
2. Preprocesamiento, comparación/selección y Accuracy, Precision, Recall, F1,
   F1 macro, ROC-AUC, Brier, coste, matrices de confusión y ROC.
3. Probabilidades, entropía, histograma, boxplot, accuracy por intervalo,
   medias, Mann–Whitney exploratorio y AUC de detección de errores.
4. Dos umbrales, junior/senior simulados, responsable por caso y comparación
   IA/junior/senior/colaboración. Optimización de validación se adelanta para
   congelar la política antes de abrir prueba.
5. SHAP global y nueve casos: tres aciertos con menor entropía, tres con mayor
   entropía y tres errores. Interpretación de impactos, coherencia, utilidad y
   confianza por caso; control numérico de aditividad.
6. HACS, tabla de la imagen, elegibilidad por mejora, defensa y limitaciones.

## Artefactos

Cada carpeta de outputs conserva diez PNG y las siguientes evidencias principales:

| Archivo | Contenido |
|---|---|
| `tabla_entrega.csv` | Las seis columnas solicitadas en la imagen |
| `tabla_resultados.csv` | Cobertura, coste, mejora, HACS y Monte Carlo por modelo |
| `decisiones_prueba.csv` | Etiqueta, probabilidad, entropía, responsable y decisiones |
| `modelos_validacion.csv`, `modelos_prueba.csv` | Métricas de los candidatos |
| `busqueda_umbrales_validacion.csv` | Todas las políticas candidatas |
| `politicas_congeladas.json` | Umbrales y justificación cuantitativa |
| `simulaciones_prueba.csv` | 500 réplicas por modelo |
| `bootstrap_mejora.csv` | 1,000 réplicas pareadas del modelo seleccionado |
| `particiones.csv` | Índices y etiquetas de las particiones |
| `casos_shap.csv`, `shap_valores.csv` | Nueve casos y atribuciones auditables |
| `discusion_resultados.md` | Discusión generada a partir de esta ejecución |
| `modelo_seleccionado.joblib` | Preprocesador, modelo y política |
| `run_summary.json`, `manifest.json` | Estado, versiones, hardware, tamaños y hashes |

Los modelos se entrenan en CPU porque el dataset es pequeño. Kaggle ejecuta en
una sesión Tesla T4 y registra las dos tarjetas disponibles; no se afirma haber
acelerado entrenamiento con GPU.

## Reproducción

Desde la raíz del repositorio, usando el Python de `.venv`:

```powershell
$env:PYTHONUTF8='1'
.venv/Scripts/python.exe PlanIA/S13/local/run_local.py
.venv/Scripts/python.exe PlanIA/S13/validate_outputs.py
```

`local/requirements.txt` registra las versiones del entorno local. Las versiones
remotas constan en `kaggle/v1/outputs/run_summary.json`. Abrir el notebook en
Jupyter y ejecutar todas sus celdas también funciona. El runner local liga el
kernel al Python activo y genera HTML sin depender del kernel base de la máquina.

Para una nueva revisión remota del experimento v1:

```powershell
Set-Location PlanIA/S13/kaggle/v1/input
kaggle kernels push -p . --accelerator NvidiaTeslaT4
kaggle kernels status jeffreyamc/hack11-human-ai-credit-v1
kaggle kernels output jeffreyamc/hack11-human-ai-credit-v1 -p ../outputs --force
```

El CLI omite el archivo interno `__notebook__.ipynb`. Se recupera el notebook
ejecutado con el SDK oficial y la versión explícita mediante
`kaggle/download_executed.py`; su evidencia queda en `notebook_download.json`.
No hay credenciales incluidas en el repositorio ni dentro de los notebooks.

Ejemplo de recuperación usando el Python que tiene instalado el SDK Kaggle:

```powershell
python PlanIA/S13/kaggle/download_executed.py --kernel jeffreyamc/hack11-human-ai-credit-v1 --version 3 --output PlanIA/S13/kaggle/v1/outputs/hack11_v1_ejecutado.ipynb --html-python .venv/Scripts/python.exe
```

En este equipo, el SDK está en el Python del entorno de la herramienta Kaggle de
uv; `python` base no lo tiene. Puede consultarse con `Get-Command kaggle` o usarse
un entorno propio que tenga `kaggle` instalado.

`build_notebooks.py` conserva la plantilla compartida. **Regenerar sobrescribe
los notebooks y borra sus salidas guardadas**; se usa sólo para editar la plantilla
y después volver a ejecutar ambos entornos.

## Fuente y atribución

Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository.
[DOI 10.24432/C5NC77](https://doi.org/10.24432/C5NC77).
[Licencia CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
El diccionario y la matriz de costes están en `data/readme.md` y `data/german.doc`.
Las seis partes y HACS proceden de la guía docente local; su texto extraído y la
página con la fórmula se conservan en `references/`.
