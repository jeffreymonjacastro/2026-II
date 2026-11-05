### Resultado observado del modelo defendido: Random Forest

Se eligió por F1 de validación (0.5298), sin usar prueba.
En prueba, F1 IA = **0.5874**, F1 Humano–IA = **0.5915**,
ΔF1 = **+0.0041** (+0.41 puntos porcentuales) y HACS = **0.7469**.
La realización reportada **supera** la IA sola; elegible por mejora: **True**.

La política A=0.6930, B=0.6930 nats automatiza **98.0%**:
196 IA, 0 junior y 4 senior, de 200 casos.
Fue seleccionada por HACS con guardarraíles de mejora en validación; no se fuerza
junior si la búsqueda prefiere A=B. Coste relativo IA **131** frente a
colaborativo **130** (no incluye coste laboral).

En 500 simulaciones, F1 colaborativo medio = **0.5903**,
intervalo Monte Carlo central 95% = [0.5816, 0.5915],
y fracción de réplicas con mejora = **83.6%**.
El intervalo bootstrap pareado de ΔF1 en réplica 0 es [0.0000, 0.0141].
Si contiene cero, la mejora de esa muestra tiene incertidumbre material.

Entropía media de aciertos **0.5954**, de errores **0.6568**;
AUC de detección de errores **0.7180** y p exploratorio
Mann–Whitney **1.193e-06**. La utilidad de entropía depende de estos
valores: no asumimos que toda equivocación sea incierta ni que toda certeza acierte.

**Appropriate reliance:** delegar según incertidumbre busca combinar automatización
con revisión; el junior puede empeorar una decisión y un senior también puede fallar.
Los beneficios observados dependen de las accuracies humanas **supuestas**, no son
evidencia de rendimiento humano real. No hemos medido calibración completa,
fatiga, dependencia humano–IA, tiempos ni equidad por subgrupos. El dataset es
histórico y pequeño; una implementación real requiere validación externa y estudio
humano antes de sostener beneficios operativos. SHAP hace la salida inspeccionable,
pero no demuestra que la asistencia explicativa causó la mejora.
