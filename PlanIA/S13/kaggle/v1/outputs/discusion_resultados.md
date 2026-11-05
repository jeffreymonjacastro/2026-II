### Resultado observado del modelo defendido: CatBoost

Se eligió por F1 de validación (0.4615), sin usar prueba.
En prueba, F1 IA = **0.5946**, F1 Humano–IA = **0.8850**,
ΔF1 = **+0.2904** (+29.04 puntos porcentuales) y HACS = **0.7350**.
La realización reportada **supera** la IA sola; elegible por mejora: **True**.

La política A=0.5058, B=0.5058 nats automatiza **51.0%**:
102 IA, 0 junior y 98 senior, de 200 casos.
Fue seleccionada por HACS con guardarraíles de mejora en validación; no se fuerza
junior si la búsqueda prefiere A=B. Coste relativo IA **153** frente a
colaborativo **53** (no incluye coste laboral).

En 500 simulaciones, F1 colaborativo medio = **0.8435**,
intervalo Monte Carlo central 95% = [0.8036, 0.8750],
y fracción de réplicas con mejora = **100.0%**.
El intervalo bootstrap pareado de ΔF1 en réplica 0 es [0.2040, 0.3881].
Si contiene cero, la mejora de esa muestra tiene incertidumbre material.

Entropía media de aciertos **0.4453**, de errores **0.5566**;
AUC de detección de errores **0.6844** y p exploratorio
Mann–Whitney **0.0001683**. La utilidad de entropía depende de estos
valores: no asumimos que toda equivocación sea incierta ni que toda certeza acierte.

**Appropriate reliance:** delegar según incertidumbre busca combinar automatización
con revisión; el junior puede empeorar una decisión y un senior también puede fallar.
Los beneficios observados dependen de las accuracies humanas **supuestas**, no son
evidencia de rendimiento humano real. No hemos medido calibración completa,
fatiga, dependencia humano–IA, tiempos ni equidad por subgrupos. El dataset es
histórico y pequeño; una implementación real requiere validación externa y estudio
humano antes de sostener beneficios operativos. SHAP hace la salida inspeccionable,
pero no demuestra que la asistencia explicativa causó la mejora.
