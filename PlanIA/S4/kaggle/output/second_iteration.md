# Segunda iteracion - resultados ejecutados

## Decision oficial

- Modelo: **CatBoost**, elegido por profit OOF sobre train.
- Calibracion: **isotonic**.
- Profit secuencial test: **S/26,660**.
- Presupuesto final: **S/27,660**.
- Intervenciones ejecutadas: **914** de 914 planeadas.
- Detencion por falta de presupuesto: **no**.
- Supera S/16 080 historico: **si**; comparacion no equivalente porque cambio la restriccion.

## Calidad predictiva en test

- ROC-AUC: 0.8440
- PR-AUC: 0.6388
- Brier: 0.1362

## Diagnostico retrospectivo

Ganador mirando test: **CatBoost**. Este dato es **retrospectivo, no valido para seleccion** y no sustituye al modelo oficial.

## Metodo

Split 80/20 estratificado, semilla 42. Seleccion y calibracion mediante predicciones OOF de train. Umbral economico `p(churn) > 0.10`; ranking descendente con desempate por `customerID`. Cada intervencion resta S/10 y cada exito suma S/100.

## Caveats

- El resultado historico S/16 080 uso Top-20%; sirve como referencia, no como comparacion causal.
- Operaciones GPU pueden variar levemente pese a semilla fija.
- Elegir el ganador por test seria fuga; por eso solo se reporta retrospectivamente.
