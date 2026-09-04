# CatBoost final — Hackathon 3

Notebook reducido al mejor modelo de `churn_decision_hackathon_v2.ipynb`.
Solo entrena **CatBoost en GPU**, calibra sus probabilidades y ejecuta la
estrategia de presupuesto secuencial.

## Criterio de aceptación

- Presupuesto inicial: **S/1 000**.
- Presupuesto final mínimo: **S/27 660**.
- Profit mínimo: **S/26 660**.
- Si CatBoost produce un presupuesto mayor, se conserva la mejora.

El notebook contiene un `assert` que detiene la ejecucion si el presupuesto
final cae por debajo de S/27 660.

## Resultado validado en Kaggle

Version 3 del kernel, ejecutada en Tesla T4:

- Estado: **COMPLETE**.
- Presupuesto final: **S/27 700**.
- Profit: **S/26 700**.
- Intervenciones: **920**.
- Exitos: **359**.
- Fallos: **561**.
- El presupuesto no se agoto.

Resultado mejora el presupuesto anterior de S/27 660 en **S/40**.

## Metodo

1. Divide 80/20 con muestreo estratificado y semilla 42.
2. Genera probabilidades out-of-fold con cinco folds de CatBoost.
3. Ajusta calibracion isotonic exclusivamente con las predicciones OOF.
4. Entrena CatBoost final sobre todo train.
5. Selecciona clientes con `p(churn) > 0.10`.
6. Ordena por probabilidad descendente y desempata por `customerID`.
7. Recorre la lista: resta S/10 y suma S/100 cuando `Churn=Yes`.

La etiqueta test solo se revela dentro del simulador, despues de congelar la
seleccion y el orden.

## Ejecucion en Kaggle

Dataset privado: `jeffreyamc/telco-churn-s4`.

Kernel privado: `jeffreyamc/churn-catboost-final-s4`.

```powershell
kaggle kernels push -p . --accelerator NvidiaTeslaT4
kaggle kernels status jeffreyamc/churn-catboost-final-s4
kaggle kernels output jeffreyamc/churn-catboost-final-s4 -p output --force
```

El notebook exige una Tesla T4 mediante `nvidia-smi` y configura CatBoost con
`task_type="GPU"` y `devices="0"`.

## Archivos

- `churn_catboost_final.ipynb`: pipeline reproducible completo.
- `kernel-metadata.json`: configuracion del kernel privado con T4.
- `output/run_summary.json`: resumen de la ejecucion validada.
- `output/intervention_ledger.csv`: trazabilidad cliente por cliente.

CatBoost GPU puede tener variaciones numericas minimas. El `assert` rechaza
regresiones, pero permite conservar mejoras.
