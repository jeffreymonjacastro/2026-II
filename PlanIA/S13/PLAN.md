# Hackatón 11 — v1

## Alcance autorizado

Dos notebooks completos, con las seis partes y actividades del PDF. Kaggle:
XGBoost, LightGBM y CatBoost, dataset privado y ejecución Tesla T4. Local:
copia del protocolo con Random Forest y Logistic Regression y ejecución local.

## Decisiones metodológicas

- Datos originales `german.data`: 1,000 filas, 20 atributos; clase positiva = mal
  crédito (código original 2). Coste adicional = FP + 5 FN.
- Split estratificado fijo 60/20/20. Preprocesamiento ajustado únicamente con
  entrenamiento; selección del modelo por F1 de validación. Sin reajustar con
  validación después de elegir umbrales, para preservar sus probabilidades.
- Entropía binaria con logaritmo natural, rango [0, ln(2)].
- Junior 70% y senior 95%: errores Bernoulli simulados, independientes de la
  clase. Son supuestos pedagógicos, no resultados de personas reales.
- Búsqueda de dos umbrales en validación con 200 réplicas comunes; optimizar
  HACS entre políticas con mejora media F1 >= 0.01 y probabilidad Monte Carlo
  de mejora >= 0.80. Si no existen, registrar explícitamente ese resultado.
- Prueba: política congelada; 500 réplicas con semillas nuevas. Reportar una
  ejecución reproducible y media/intervalo Monte Carlo por separado; bootstrap
  pareado de casos para la incertidumbre muestral de la mejora de esa ejecución.
- F1 binario principal, positivo = mal crédito; F1 macro adicional. Mejora =
  F1 colaborativo - F1 IA, en escala [−1,1], y puntos porcentuales explícitos.
- HACS exacto del PDF: 0.6 F1 + 0.2 Coverage + 0.2 Efficiency. Coverage y
  Efficiency coinciden porque cada solicitud tiene exactamente un responsable.
- SHAP para el modelo elegido: global y nueve casos, con explicaciones de
  impacto, coherencia, utilidad y límites. No atribuir causalidad ni mejora
  humana a SHAP sin un experimento adicional.

## Verificación y entrega

1. Subir copia exacta y atribuida del dataset a Kaggle privado.
2. Generar ambos notebooks desde una plantilla común autocontenida.
3. Ejecutar local y Kaggle en paralelo, recuperar notebook remoto ejecutado,
   CSV, imágenes, manifiestos y resumen de ejecución.
4. Recalcular F1, mejora, cobertura, HACS y coste desde predicciones; validar
   split, archivos, semillas y celdas; inspeccionar figuras y HTML renderizado.
5. README con rutas, resultados observados, reproducción y limitaciones.
