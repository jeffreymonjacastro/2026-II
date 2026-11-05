# v2 — F1 de buen crédito y optimización conjunta de HACS

## Requisitos

- Notebook Kaggle v2 completo, con las seis partes de la guía y tres finalistas.
- Clase positiva `1 = buen crédito` (código original 1); `0 = mal crédito`.
- Mejora estricta `F1 colaborativo − F1 IA > 0.01` (>1 punto porcentual).
  Reportar también porcentaje relativo. No fabricar ni garantizar un resultado
  de prueba antes de observarlo.
- Mejorar HACS con selección conjunta de modelo y política en validación.

## Diseño fijado antes de evaluar v2 en prueba

1. Reutilizar dataset privado y las particiones de v1 (600/200/200), conservando
   la estratificación en la codificación original para no cambiar índices.
2. Comparar cinco familias: XGBoost, LightGBM, CatBoost, Random Forest y Logistic
   Regression. Búsqueda pequeña de hiperparámetros por CV estratificada de cinco
   folds dentro de entrenamiento; preprocesamiento dentro de cada pipeline.
3. Elegir el umbral binario del modelo con probabilidades OOF de entrenamiento
   para F1 de buen crédito. No debilitar artificialmente el baseline IA.
4. Simular junior 70% y senior 95%, como v1. No elevar acierto humano para ganar.
   Seleccionar A/B de entropía únicamente en validación, con 500 simulaciones:
   ΔF1 medio >=0.02 y percentil 10 de ΔF1 >0.01. Son guardarraíles preventivos.
   Elegir máximo HACS entre políticas que cumplan; registrar si no hay elegibles.
5. Ordenar familias por HACS colaborativo medio de validación y congelar las tres
   primeras elegibles y sus políticas. Sólo esas tres se evalúan en prueba.
6. Prueba: réplica 0 prefijada y 1,000 simulaciones nuevas; no elegir semillas ni
   revisar umbrales usando etiquetas de prueba. La mejora >0.01 se audita tanto
   en la réplica reportada como en la media Monte Carlo.
7. SHAP, nueve casos, incertidumbre, ROC, confusión, costos y fórmulas completos.
   Con `positivo = bueno`, coste original = `5 FP + FN`.
8. Comparar v1/v2 también con idéntica clase positiva: recodificar predicciones
   guardadas de v1 para distinguir cambio de métrica de mejora experimental.

## Límites

La prueba fue vista durante v1; no es un holdout nuevo. La configuración v2 se
elige sin consultar sus resultados v2, pero una afirmación de generalización
fuerte requeriría validación externa. La mejora por simulación humana depende de
accuracies supuestas, constantes incluso en casos inciertos. HACS no diferencia
coste junior/senior, por lo que puede elegir A=B y revisión sólo senior.

## Evidencias de cierre

Kaggle COMPLETE, run_summary complete/done, notebook remoto realmente ejecutado,
fuentes coincidentes, tres finalistas seleccionados antes de prueba, F1 positivo
bueno y coste auditados desde decisiones, mejora estricta >0.01, HACS y cobertura
recalculados, CV sin acceso a validación/prueba, CSV/figuras/HTML descargados y
presentación inspeccionada. Si algún requisito numérico falla, debe reportarse
como incumplido, sin cambiar el significado de mejora ni ocultar resultados.
