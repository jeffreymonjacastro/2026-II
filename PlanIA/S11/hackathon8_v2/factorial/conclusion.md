**Configuración recomendada:** LoRA con r=1 y alpha=32, elegido por validación según la tolerancia práctica fijada antes del experimento. Su F1 medio de validación es 0.828 y el de prueba 0.891.

Al pasar de r=1 a r=2 con alpha=32, la diferencia media en validación fue -0.022. Su límite superior unilateral del 95 % fue 0.017, menor que el margen práctico de 0.02. En este tramo, con los casos y semillas medidos, los resultados no justifican aumentar el rango para buscar una ganancia de 0.02 o más.

El máximo previsto por la superficie fue r=1, alpha=32, con F1 0.807. La comprobación real con dos semillas obtuvo 0.828. El modelo de regresión orienta la búsqueda, pero no reemplaza las mediciones.

En los 128 casos nuevos, el F1 fue 0.794 para base, 0.835 para LoRA y 0.838 para DoRA, promediando las dos réplicas de los adaptadores. DoRA necesitó 2.91 veces el tiempo de entrenamiento y 1.09 veces la VRAM de LoRA. La diferencia DoRA menos LoRA en confirmación fue +0.004, con intervalo del 95 % [-0.017, +0.027].

El óptimo y la frontera corresponden a la política sintética, 256 ejemplos de entrenamiento y 32 actualizaciones. No son límites universales de LoRA. Dos semillas y una GPU compartida restringen la precisión de las conclusiones.

**Comparación ampliada:** rsLoRA obtiene F1 0.855 y LoRA+ 0.709 en los 128 casos nuevos. Entre los métodos con rango y alpha compartidos, la mayor media de validación corresponde a DoRA. Las tasas de LoRA+ se mantienen en razón B/A=16 durante los 32 pasos. Estos métodos comparten la configuración seleccionada para LoRA; no se buscaron sus óptimos por separado. Los intervalos de las tres comparaciones son exploratorios y no están ajustados por multiplicidad.

Con r=1, las escalas alpha/r y alpha/sqrt(r) coinciden. Por eso las diferencias observadas entre LoRA y rsLoRA en ese rango no se atribuyen al reescalamiento. Se fijaron las semillas, pero no se forzó el uso exclusivo de kernels deterministas de CUDA. Se agregó una comparación duplicada de rsLoRA con r=32 frente al LoRA correspondiente del factorial, donde las escalas sí difieren. Sus resultados se conservan en rslora_control_summary.csv y no cambian la configuración compartida ni la selección con confirmación.

La mayor media descriptiva en la confirmación corresponde a rsLoRA (0.855), mientras que validación había elegido DoRA. No se cambia la selección usando este conjunto nuevo. El cambio de orden muestra la incertidumbre de comparar métodos con particiones pequeñas y dos semillas.