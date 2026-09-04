# MLP final — Hackathon 3

Notebook reducido a la MLP usada en la segunda iteracion. Entrena solo la red
neuronal, calibra probabilidades y aplica la politica de presupuesto secuencial.

## Criterio de aceptación

- Presupuesto inicial: **S/1 000**.
- Presupuesto final mínimo: **S/27 550**.
- Profit mínimo: **S/26 550**.
- Si una ejecucion mejora el resultado, se conserva.

El notebook tiene un `assert` que falla si el presupuesto final es menor a
S/27 550.

## Resultado validado en Kaggle

Versión 1 del kernel, ejecutada en Tesla T4:

- Estado: **COMPLETE**.
- Presupuesto final: **S/27 550**.
- Profit: **S/26 550**.
- Intervenciones: **875**.
- Éxitos: **353**.
- Fallos: **522**.
- El presupuesto no se agotó.

## Arquitectura y método

```text
inputs -> Dense(64, ReLU) -> Dropout(0.2) -> Dense(32, ReLU)
       -> Dropout(0.1) -> Dense(1, sigmoid)
```

El entrenamiento usa `BCEWithLogitsLoss` y aplica `sigmoid` al obtener
probabilidades. Usa One-Hot Encoding en variables categóricas,
StandardScaler en numéricas, split 80/20 estratificado con semilla 42 y cinco
folds OOF. Cada fold usa `EarlyStopping` por `val_loss`; el modelo final usa
la mediana de los mejores epochs.

Las probabilidades se calibran con isotonic sobre scores OOF. Se interviene si
`p(churn) > 0.10`, ordenando descendentemente y usando `customerID` como
desempate. Cada intervención resta S/10; un churn exitoso suma S/100.

## Ejecución Kaggle

Dataset privado: `jeffreyamc/telco-churn-s4`.

Kernel privado: `jeffreyamc/churn-mlp-final-s4`.

```powershell
kaggle kernels push -p . --accelerator NvidiaTeslaT4
kaggle kernels status jeffreyamc/churn-mlp-final-s4
kaggle kernels output jeffreyamc/churn-mlp-final-s4 -p output --force
```

El notebook exige una Tesla T4 y usa PyTorch en `cuda:0`.

## Archivos

- `churn_mlp_final.ipynb`: implementación MLP reproducible.
- `kernel-metadata.json`: kernel privado con T4.
- `output/run_summary.json`: resultado validado.
- `output/intervention_ledger.csv`: trazabilidad de cada intervención.

Operaciones CUDA pueden variar mínimamente entre ejecuciones. El contrato de
no regresión impide aceptar un presupuesto menor a S/27 550.
