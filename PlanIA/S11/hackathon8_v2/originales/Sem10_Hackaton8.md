**Hackaton 8: Refinamiento de LLMs**

Docente: Robert Buleje

**Motivación**

Durante la semana de agentes IA exploramos el uso de LLM para el desarrollo de agentes unimodelo y multimodales. Estos agentes, se recomiendan para tareas que necesitas un razonamiento para tomar una decisión y no siempre sea posible aplicar un conjunto estricto de reglas como un algoritmo clásico. Ejemplos de estas tareas tenemos: clasificación de tickets de atención de acuerdo a la urgencia expresada por el usuario, extracción de información de data no estructurada, clasificación de sentimientos, atención al cliente, entre otras aplicaciones.

Si bien, podemos personalizar al agente mediante prompting engineering y las estructuras que ofrecen los diferentes frameworks para encaminar su actuar; de todas formas existe un sesgo asociado al entrenamiento inicial (conocimiento general) de los LLMs que se utilizan para estos agentes. Si tuvieran que realizar una comparación con lo cotidiano, diseñar e implementar un agente con un LLM de un tercero equivaldría a comprar ropa de stock (*ready to wear*). Si bien funciona, no fue pensado estrictamente para nosotros.

Continuando en el mundo de la sastrería, estimados futuros maestros sartoriales de las Ciencia de Datos y Ciencias de la Computación, existe una alternativa, que es la máxima expresión de esta rama del conocimiento humano, una prenda confeccionada bajo una filosofía *bespoke*, un patrón desde cero para un irrepetible individuo. Sin embargo, confeccionar un LLM desde cero, representa costos y tiempos, que no siempre todos nuestros clientes se pueden permitir. 

El mundo está lleno de aquellos irrazonables que se resisten a seguir el rumbo de lo cotidiano y mundano, aquellos que su ojo magnifica el tamaño de los idea, salvo las dificultades (referencia a Julio Verne). Volviendo a los sartorial, existe una alternativa intermedia que rescata lo mejor del *ready to wear* y lo *bespoke*, que se conoce como *made to measure*. Cambiando nuevamente de espacio de conocimiento, los LoRA y herramientas posteriores, nos permiten personalizar, con algunas limitaciones, un LLM para que sea más eficiente en resolver nuestra caústica en particular. Nos evita la titánica tarea de diseñar nuestro patrón desde cero y confeccionar a mano con millones de ejemplos y meses de trabajo; a poder adaptar a partir de un patrón existe en tiempos razonables y cómputo alcanzable.

**Instrucciones**

Cada equipo debe elegir un LLM que quepa en la tarjeta gráfica que utilizarán para el entrenamiento, ya sea por COLAB o utilizando el Khipu de la UTEC. Se deben cumplir los siguiente criterios.

- No puede ser el mismo LLM que el docente utilizó durante la clase para la demostración del LoRA. Tampoco debe ser uno de la misma familia (mismo modelo, diferente número de parámetros).  
- Se permite que máximo 2 equipos utilicen la misma familia de modelos, pero se agradecería que traten de ser todos distintos para que aporte riqueza y variedad a las exposiciones.  
- De preferencia utilizar modelos en español, para que este hackaton les sirva para trasladarlo a sus prácticas profesionales. 

Cada equipo debe elegir una tarea para evaluar objetivamente el desempeño del modelo, antes y después de aplicar un LoRA. Algunos ejemplos:

- Clasificación de sentimientos o emociones de frases.  
- Clasificación y derivación de tickets de soporte técnico.  
- Priorización de tareas de acuerdo a múltiples criterios de urgencia.  
- Incluir de preferencia salidas estructuradas como json, xml o markdown.  
- Evitar problemas de clasificación binaria, mínimo 4 clases.  
- Se valorará la complejidad del problema a resolver con el agente.  
- No es necesario que sea multiagente, nos enfocaremos en el desempeño del LLM con una única tarea.  
- Si bien pueden utilizar LLMs como Claude, Chat GPT o Gemini para generar el dataset de entrenamiento, se recomienda que al menos el 20% del dataset sea escrito a mano por los estudiantes.

Finalmente, el equipo debe realizar una mini investigación (en mi opinión profesional, este trabajo, siguiendo una metodología más rigurosa, tendría potencial como tesis para el grado de bachiller; si resuelven una problemática real, si podría calificar para tesis de titulación). Se sugiere realizar lo siguiente.

- Plantear un diseño experimental factorial (video de referencia: [Diseños factoriales. Diseños de investigación en Psicología 8/10](https://www.youtube.com/watch?v=EYKfpKXErtQ)). por ejemplo. 3 niveles de r y 2 niveles de alpha, esto daría un total de 6 experimentos (salvo realicen los experimentos por duplicado o triplicado, diferente semilla para cada repetición). El diseño experimental que decidan seguir debe poder contribuir a inferior en que punto no es significativo ni  provechoso seguir invirtiendo recursos por mejorar marginales en el desempeño del LoRA. Mi recomendación es primero probar extremos en los hiper-parámetros o parámetros que decidan variar y luego a partir de esos resultados, plantear su diseño experimental final.  
- Se deben evaluar varios aspectos en cada experimento: la métrica de desempeño del modelo en la tarea, los requisitos computacionales (cuanta VRAM se necesita para ejecutar el entrenamiento del LoRA), el tiempo de entrenamiento, tiempo de inferencia, etc. Para evitar diferencia entre los resultados, utilizar siempre los mismo componentes computacionales, por ejemplo, solo COLAB o un modelo específico de GPU del Khipu.  
- Podrían hacer una regresión lineal simple, polinómica o no lineal para hallar el punto óptimo entre desempeño e hiper-parámetros o las variables que decidieron utilizar en el diseño experimental.

Documento con las elecciones de los equipos: [Equipos Hackaton8 Temas](https://docs.google.com/spreadsheets/d/1KZq4sFJZe_-Tx0MMaDKdLstGzx68KfzhNAD6Isd1XeI/edit?usp=sharing)

Finalmente, los equipos deben comparar sus mejores LoRAs contra una tecnología más actual, también pueden replicar el mismo diseño experimental. Se sugiere a los equipos que utilicen una tecnología que esté implementada en la librería **PEFT** de Hugging Face. No se recomienda que implementen un método desde cero a partir de un paper, por un tema de tiempo y factibilidad, sin embargo, expresamente no se les prohíbe, queda a discreción del equipo.

**Nota: los notebooks de cada equipo serán socializados con todos los alumnos del curso, salvo que un equipo indique lo contrario por escrito.**

**Rúbrica de evaluación (20 puntos)**

| Criterio | Puntaje |
| :---- | :---- |
| **Selección de la problemática** 2 puntos. Se selecciona una problemática desafiante para el LLM, con varias capas de complejidad. Refleja una problemática real. 1 punto. Se selecciona una problemática simple. 0 puntos. Problemática sin métrica objetiva.   | 2 |
| **Diseño experimental** 5 puntos. El diseño experimental permite evidenciar los límites de la tecnología del LoRA. Se encuentra la frontera en dónde el LoRA no presenta mejora en el desempeño de la tarea. Permite encontrar un punto óptimo. Utiliza al menos duplicado de los experimentos. Utiliza correctamente las semillas. Una semilla por réplica (por ejemplo, si el diseño incluye 9 experimentos, la primera ronda se ejecuta con una semilla, mientras que la segunda ronda, el duplicado, se ejecuta con otra semilla). 2 puntos. Diseño experimental básico, permite una análisis objetivo, si bien es posible encontrar un óptimo, no se comprueba ese resultado teórico (fuera del dominio de experimentación).   0 puntos. No realiza un diseño experimental. | 5 |
| **Análisis de los resultados.** 5 puntos. Excelente análisis de los resultados de la experimentación. Se analiza el problema desde diferentes perspectivas: desempeño del modelo, costo computacional, coste de entrenamiento (simular costo de utilizar herramientas de paga) , entre otras perspectivas. Llega a una conclusión sólida. 2 puntos. Análisis básico y elemental de los resultados. Se enfoca en una dimensión del problema.  0 puntos. No realiza un análisis, se limita a presentar los resultados. | 5 |
| **Notebook** 2 puntos. El notebook del entrenamiento y evaluación son reproducibles, utiliza POO u otra herramienta para colocar de forma ordenada todos los parámetros. Está comentado. 1 punto. Se evidencia desorden, pero sigue siendo legible. **0 puntos. Ilegible, obtener cero en este criterio invalida el trabajo y se coloca cero inmediatamente.**  | 2 |
| **Exposición**. 2 puntos. El equipo construye una narrativa sobre sus experiencias ejecutando el hackaton y lo presenta de forma efectiva en un PPT ejecutivo. Preguntas adecuadamente sustentadas. 1 punto. Exposición regular con puntos flojos. 0 puntos. Exposición improvisada. | 2 |
| **Comparación de tecnologías** 4 puntos. El equipo utiliza un tecnología que contrasta con los resultados del LoRA. Justifica su decisión. Explica correctamente el fundamento de la tecnología. Realiza un análisis objetivo de los resultados. 2 puntos.  0 puntos.  | 4 |

