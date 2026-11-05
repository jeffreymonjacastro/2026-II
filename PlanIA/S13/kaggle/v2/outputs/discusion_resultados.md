### Resultado observado del modelo defendido: Random Forest

En la v2, la IA resolvió automáticamente entre el 87% y el 92% de los casos. Sólo los casos con mayor incertidumbre se enviaron al analista senior simulado, con un 95% de probabilidad de acierto. Esta revisión permitió corregir errores y mejorar el F1 sin recurrir demasiado al humano.

Ese equilibrio explica el HACS obtenido: el puntaje premia tanto el F1 como la automatización. Por ejemplo, CatBoost automatizó el 90% de las decisiones y su F1 pasó de 0.8487 a 0.8805, una mejora de 3.19 puntos porcentuales. Así alcanzó un HACS de 0.8883, el mayor observado en prueba.

La política no utilizó al junior porque HACS penaliza igual una revisión junior que una senior. Como el senior tiene mayor acierto supuesto, la búsqueda prefirió asignarle los casos que necesitaban revisión.

Los tres modelos superaron la mejora exigida de un punto porcentual. Estos resultados corresponden al F1 de buen crédito y dependen de las capacidades humanas simuladas; no representan mediciones de analistas reales.
