# Decisiones técnicas — Hackathon 3: Modelado de Churn

Este documento justifica, sección por sección (alineado a la rúbrica de
`hack 3 (2).md`), cada decisión técnica tomada en
`churn_decision_hackathon.ipynb`. Los números reportados aquí son los
mismos que produce el notebook al ejecutarlo de punta a punta.

## 1. Planteamiento del problema

- **Variable objetivo:** `Churn` (Yes/No), codificada como `ChurnFlag`
  (1/0). Es la única variable en el dataset que representa directamente
  el evento de negocio que se quiere anticipar.
- **Tipo de modelo:** un clasificador binario cuya salida útil es la
  **probabilidad** de churn (`predict_proba`), no la clase dura. La
  empresa no necesita saber "se va / no se va" para todos los clientes:
  necesita un **ranking de riesgo**, porque solo puede actuar sobre el
  20% de la base. Un modelo que solo devuelva 0/1 no permite priorizar
  dentro de ese 20%.
- **Métricas de modelo:** ROC-AUC, PR-AUC, Precision/Recall@20% y Brier
  score (calibración). Se evita apoyarse en accuracy porque, con ~26.5%
  de churn, un modelo trivial que siempre prediga "No" ya obtiene ~73.5%
  de accuracy sin aportar ningún valor de negocio.
- **Métrica de negocio (la que realmente decide):**
  `Profit = 100 * retenidos_correctamente − 10 * intervenidos`. Esta es
  la métrica que finalmente se optimiza en la sección de decisión, por
  encima de cualquier métrica puramente estadística del modelo.

## 2. Análisis de datos

- Dataset: 7043 clientes, 21 columnas, sin duplicados de `customerID`.
- **Limpieza de `TotalCharges`:** venía tipada como texto porque 11 filas
  tienen el valor `" "` (string vacío). Se verificó (con un `assert` en
  el notebook) que esas 11 filas coinciden exactamente con clientes de
  `tenure == 0`, es decir, clientes tan nuevos que aún no se les generó
  ninguna facturación acumulada. Se convirtieron a `NaN` y se imputaron
  con `0.0`, en vez de eliminar las filas, para no perder esos 11
  registros (clientes de tenure 0 suelen ser un segmento de riesgo).
- **Desbalance de clases:** 26.5% de los clientes hacen churn (73.5% no).
  Es un desbalance moderado — no extremo, pero sí suficiente para exigir
  métricas más allá de accuracy y para justificar el uso de pesos de
  clase en el entrenamiento (ver sección 4).
- **Señal por variable:** `Contract` (mes a mes vs. contratos anuales/
  bianuales) y `tenure` bajo muestran la asociación más fuerte con
  churn, seguidas de `InternetService = Fiber optic` y la ausencia de
  `OnlineSecurity` / `TechSupport`. Es un patrón consistente con la
  intuición de negocio: clientes sin compromiso contractual y sin
  servicios "de retención" (soporte, seguridad) son más propensos a
  irse.

## 3. Diseño de pipeline

Construido desde cero con `ColumnTransformer` + `Pipeline` de
scikit-learn (sin AutoML), para mantener control total y trazabilidad
sobre cada transformación:

- **Numéricas** (`tenure`, `MonthlyCharges`, `TotalCharges`): imputación
  por mediana (robusta a outliers, aunque en la práctica no quedaron
  nulos tras la limpieza) + `StandardScaler`. El escalado es necesario
  para que la regresión logística no esté dominada por las variables de
  mayor magnitud (`TotalCharges` vs. variables 0/1); a los modelos de
  árbol (Random Forest, XGBoost) el escalado no les afecta, así que
  compartir el mismo preprocesador para los tres modelos no introduce
  ningún costo.
- **Categóricas** (el resto de columnas, incluyendo `SeniorCitizen`
  tratada como categórica binaria): `OneHotEncoder(handle_unknown=
  'ignore')`. El `handle_unknown='ignore'` evita que el pipeline falle
  si en producción apareciera una categoría no vista en entrenamiento.
- **División train/test:** 80/20 **estratificada** por `Churn`, para que
  ambos conjuntos preserven la misma tasa de churn (~26.5%) y las
  métricas de test sean representativas. Verificado en el notebook: la
  diferencia entre la tasa de churn de train y test es menor a 2 puntos
  porcentuales.
- Todo el preprocesamiento vive **dentro** del `Pipeline`, ajustado
  únicamente con `X_train`, para evitar fuga de información hacia test.

## 4. Modelado

Se entrenaron y compararon tres modelos sobre el mismo pipeline y split:

| Modelo | Por qué se eligió | Ventajas | Limitaciones |
|---|---|---|---|
| Regresión Logística (`class_weight='balanced'`) | Baseline interpretable | Coeficientes explicables al negocio, rápida de entrenar | No captura interacciones no lineales |
| Random Forest (`class_weight='balanced'`) | Captura no linealidades sin mucho tuning | Robusto a outliers/escala, da importancia de variables | Menos interpretable, riesgo de sobreajuste con árboles muy profundos |
| XGBoost (`scale_pos_weight` = ratio negativos/positivos) | Suele dar el mejor ranking de probabilidad en datos tabulares | Regularización incorporada, alto poder predictivo | Más hiperparámetros, algo menos interpretable "out of the box" |

**Manejo del desbalance:** se usaron pesos de clase (`class_weight` /
`scale_pos_weight`) en lugar de oversampling (SMOTE). La decisión final
de negocio depende de un **ranking de probabilidades** razonablemente
calibradas; generar filas sintéticas (SMOTE) puede distorsionar esas
probabilidades, mientras que los pesos de clase solo ajustan la función
de pérdida del entrenamiento sin inventar datos.

## 5. Evaluación

Resultados en test (1409 clientes, 374 con churn real = 26.5%):

| Modelo | ROC-AUC | PR-AUC | Precision@20% | Recall@20% | Brier |
|---|---|---|---|---|---|
| Logistic Regression | 0.8415 | 0.6325 | 0.6596 | 0.4973 | 0.1687 |
| Random Forest | 0.8420 | 0.6509 | 0.6773 | 0.5107 | 0.1625 |
| **XGBoost** | 0.8413 | **0.6556** | 0.6702 | 0.5053 | **0.1611** |

Los tres modelos quedan muy cerca en ROC-AUC (~0.84), lo cual es
esperable dado que comparten el mismo pipeline y el dataset no tiene
señales extremadamente no lineales. La diferencia relevante está en
**PR-AUC**, donde XGBoost gana (0.6556 vs. 0.6509 de Random Forest y
0.6325 de Regresión Logística) y además tiene el mejor Brier score
(probabilidades mejor calibradas). Por eso se selecciona **XGBoost**
como modelo final: es el que mejor rankea a la minoría de clientes que
efectivamente hace churn, que es exactamente la capacidad que explota la
estrategia de decisión de la siguiente sección — no se eligió por
accuracy ni por ROC-AUC.

## 6. Estrategia de decisión

Con el presupuesto de intervenir al 20% de los 1409 clientes de test
(282 clientes), usando las probabilidades de XGBoost:

| Estrategia | Clientes intervenidos | Retenidos correctamente | Profit |
|---|---|---|---|
| **Modelo (XGBoost) Top-20%** | 282 | 189 | **16 080** |
| Aleatorio (20%) | 282 | 74 | 4 580 |
| Oráculo (óptimo con presupuesto 20%) | 282 | 282 | 25 380 |

El modelo captura **189 de los 374 churners reales** del test
interviniendo únicamente 282 clientes, generando un profit de **16 080**
— **3.5× más** que la estrategia aleatoria con el mismo presupuesto
(4 580), y alcanzando el **63% del techo teórico** marcado por el oráculo
(25 380, el máximo posible con exactamente 282 intervenciones, ya que
hay 374 churners reales y el presupuesto solo alcanza para 282 de
ellos).

**Top-K vs. threshold:** se usó Top-K = 20% como estrategia principal
porque garantiza el cumplimiento exacto de la restricción de recursos
sin importar cómo se distribuyan las probabilidades del modelo. Como
verificación, se calculó el umbral de probabilidad equivalente (cuantil
80 de los scores = **0.7395**): intervenir con ese umbral fijo produce
exactamente los mismos 282 clientes y el mismo profit (16 080),
confirmando que ambas formulaciones son equivalentes una vez que el
umbral se calibra al presupuesto disponible — pero solo Top-K garantiza
esa equivalencia de forma directa sin tener que buscar el umbral.

**¿Es 20% el punto óptimo?** El barrido de profit vs. % de clientes
intervenidos (sin la restricción de negocio) muestra que el profit
sigue subiendo más allá del 20%, alcanzando su máximo (**26 540**) cerca
del **65%** de la base. Esto significa que, en términos puramente de
profit sin restricciones, a la empresa le convendría intervenir a más
del 20% de sus clientes — el 20% actual es una restricción de recursos
operativa (capacidad del equipo de retención), no el punto matemáticamente
óptimo de la curva de profit. Esta es información directamente
accionable: si la empresa pudiera ampliar su capacidad de intervención
hacia ~65%, el profit marginal seguiría siendo positivo.

## Limitaciones y trabajo futuro

- No se realizó una búsqueda extensiva de hiperparámetros (grid/random
  search); se usó una configuración razonable y justificada por modelo,
  ya que el foco del hackathon es la decisión de negocio, no el tuning
  exhaustivo.
- La calibración de probabilidades (Brier ≈ 0.16) podría mejorarse con
  técnicas como Platt scaling o isotonic regression si se necesitara una
  interpretación más literal de "probabilidad de churn" fuera del
  contexto de ranking.
- El análisis de costo/beneficio asume que el costo de intervención (10)
  y el valor de retención (100) son uniformes para todos los clientes;
  en un escenario real, ambos podrían variar según el valor del cliente
  (p. ej. `MonthlyCharges` o `TotalCharges`), lo cual cambiaría el
  ranking óptimo de intervención.
