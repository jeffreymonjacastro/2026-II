# Hackatón 11 — v2: F1 de buen crédito y HACS

**Kaggle COMPLETE**, revisión remota 1. Los tres finalistas superan una mejora
estricta de **un punto porcentual** de F1 frente a su IA sola, tanto en la réplica
reportada como en la media de 1,000 simulaciones humanas.

- [Notebook en Kaggle](https://www.kaggle.com/code/jeffreyamc/hack11-human-ai-credit-v2-good-credit).
- Notebook con resultados: `outputs/hack11_v2_ejecutado.ipynb`.
- Vista HTML: `outputs/hack11_v2.html`.
- Fuente editable para subir: `input/hack11_v2.ipynb`.
- Tabla solicitada: `outputs/tabla_entrega.csv`, en **Parte 6 → Tabla de entrega y defensa**.
- Auditoría independiente: `validation_report.json` y `validate_v2.py`.
- Comparación histórica con igual clase positiva: `comparacion_v1_v2_misma_clase.csv`.

## Resultados de prueba

Clase positiva **1 = buen crédito**; clase 0 = mal crédito. La mejora es
`F1 colaborativo − F1 IA`; el requisito es **>0.01**, no ≥0.01. La tabla usa la
réplica 0, prefijada antes de entrenar; no se escogió una semilla favorable.

| Modelo | F1 IA | F1 IA + humano | Mejora (puntos porcentuales) | HACS | IA | Junior | Senior |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.8387 | 0.8638 | +2.51 | 0.8863 | 184 | 0 | 16 |
| CatBoost | 0.8487 | 0.8805 | +3.19 | 0.8883 | 180 | 0 | 20 |
| XGBoost | 0.8191 | 0.8660 | +4.69 | 0.8676 | 174 | 0 | 26 |

Cada modelo evalúa las mismas 200 solicitudes. La cobertura es 92%, 90% y 87%,
respectivamente. **Random Forest es el modelo defendido**, seleccionado previamente
por HACS de validación. CatBoost obtuvo el mayor HACS puntual en prueba; eso no
cambia retrospectivamente la selección.

## Qué cambió respecto a v1

1. F1 principal de buen crédito, con F1 de mal crédito y F1 macro complementarios.
2. Cinco familias candidatas: Random Forest, CatBoost, XGBoost, Logistic Regression
   y LightGBM. Se conservan sólo los tres mejores HACS de validación como finalistas.
3. GridSearchCV de cinco folds dentro de entrenamiento, con imputación, escalamiento
   y one-hot ajustados dentro de cada pipeline/fold. Son 26 configuraciones en total.
4. Umbral binario elegido con probabilidades OOF de entrenamiento para maximizar
   F1 de IA sola; no se debilita artificialmente el baseline.
5. Política A/B de entropía seleccionada con 500 simulaciones de validación. Se
   exige mejora media ≥0.02 y percentil 10 de mejora >0.01 antes de maximizar HACS.
6. Tres finalistas congelados antes de predecir prueba; se evalúan con 1,000 nuevas
   simulaciones, sin cambiar modelos ni políticas al ver sus resultados.

HACS medio de validación: Random Forest **0.8696**, CatBoost **0.8621**,
XGBoost **0.8539**, Logistic Regression 0.8535 y LightGBM 0.8452.

Las particiones son exactamente las de v1: 600/200/200 filas. Los notebooks y
resultados de v1, incluida la variante local, se conservan.

## Fórmulas, política y humanos

```
F1 bueno      = 2 TP / (2 TP + FP + FN)
Mejora        = F1 colaborativo − F1 IA
Mejora_pp     = 100 × Mejora
Mejora_rel_%  = 100 × Mejora / F1 IA
Entropía      = −p ln(p) − (1−p) ln(1−p), p = P(buen crédito)
Coverage      = casos IA / total
Efficiency    = 1 − casos humanos / total
HACS          = 0.6 F1 colaborativo + 0.2 Coverage + 0.2 Efficiency
Coste         = 5 FP + FN
```

Como positivo significa bueno, FP aprueba a un cliente malo y cuesta cinco;
FN rechaza a uno bueno y cuesta uno. La fórmula de HACS no cambia.

Si entropía < A, decide IA; entre A y B, junior; desde B, senior. Junior acierta
con probabilidad 70% y senior con 95%, igual que v1. Son simulaciones Bernoulli,
sin participantes reales. Las políticas eligieron A=B, por lo que no asignan
casos al junior: HACS no cobra más por usar senior. No se forzó una franja junior
ni se incrementó artificialmente la accuracy humana.

## Evidencias y límites

Se conservan las seis partes del PDF, diez PNG, SHAP global y nueve casos,
matrices de confusión, ROC, incertidumbre, costes y discusión. El notebook remoto
tiene 41 celdas, 20 de código ejecutadas y cero errores. El SDK oficial recuperó
el notebook auténtico con salidas; no se ejecutaron estos modelos localmente.

`manifest.json` y `download_integrity.json` verifican los 37 artefactos remotos;
`notebook_download.json` documenta versión, endpoint y SHA-256 del notebook.
La descarga CLI se interrumpió transitoriamente; `recover_outputs.py` recuperó
los archivos faltantes sin reiniciar ni repetir el entrenamiento.

Cambiar la clase positiva eleva o reduce F1 sin cambiar las predicciones. Por eso
se incluye una comparación de v1 recodificada como bueno, además de los resultados
originales. Con igual clase positiva, el mayor HACS de v1 era **0.8571**; en v2
alcanza **0.8883**. Los tres finalistas v2 superan ese máximo histórico puntual.
La comparación puntual también contiene variación humana simulada;
no constituye una atribución causal del cambio de HACS a un componente específico.

La prueba ya se examinó en v1; no es un holdout externo nuevo. Las accuracies
humanas constantes, incluso para casos difíciles, son supuestos pedagógicos.
El Monte Carlo varía respuestas humanas sobre la misma muestra; el bootstrap
pareado varía casos de la réplica reportada, sin repetir todo el entrenamiento.
Mejorar F1 bueno/HACS no garantiza reducir el coste bancario ni mejorar equidad.

## Reproducción

Desde la raíz del repositorio, con el CLI autenticado:

```powershell
$env:PYTHONUTF8='1'
Set-Location PlanIA/S13/kaggle/v2/input
kaggle kernels push -p . --accelerator NvidiaTeslaT4
kaggle kernels status jeffreyamc/hack11-human-ai-credit-v2-good-credit
kaggle kernels output jeffreyamc/hack11-human-ai-credit-v2-good-credit -p ../outputs --force
```

Para recuperar el notebook ejecutado se reutiliza `kaggle/download_executed.py`
con `--version 1` y el slug v2; debe ejecutarse con el Python que contiene el SDK
Kaggle. Desde la raíz, la auditoría se ejecuta con:

```powershell
.venv/Scripts/python.exe PlanIA/S13/kaggle/v2/validate_v2.py
```

El notebook registra las versiones remotas. La sesión tiene Tesla T4, pero el
entrenamiento utiliza CPU porque el dataset es pequeño. No se incluyen secretos.
Completar los nombres del equipo antes de entregar. `build_v2.py` regenera sólo
la fuente v2, sin sobrescribir v1 ni la variante local.

Fuente: Hofmann (1994), [UCI German Credit](https://doi.org/10.24432/C5NC77), CC BY 4.0.
