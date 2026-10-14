### Urgencia primero
- Horas totales: 16.0 | desbalance diario (desv. est.): 1.22 h | % horas de cursos difíciles en la mañana: 17%
- Horas por día: {'Lunes': 5.0, 'Martes': 5.0, 'Miércoles': 4.0, 'Jueves': 0.0, 'Viernes': 0.0, 'Sábado': 2.0, 'Domingo': 0.0}
- Horas por curso: {'Machine Learning': 8.0, 'Estadística Inferencial': 4.0, 'Bases de Datos II': 4.0, 'Ética en IA': 0.0, 'Comunicación Efectiva': 0.0}
- VIOLACIONES DURAS (2): Curso por debajo de su mínimo obligatorio: Estadística Inferencial 4.0 h (mínimo 4.5 h, sugeridas 6 h); Curso por debajo de su mínimo obligatorio: Ética en IA 0.0 h (mínimo 1 h, sugeridas 2 h)
- Observaciones (2): Cobertura baja: Comunicación Efectiva 0.0 h de 2 h sugeridas; Solo 17% de las horas de cursos difíciles son en la mañana
- Sugerencias de corrección: A Estadística Inferencial le faltan 0.5 h para su mínimo: asígnale franjas libres permitidas (F09, F10, F11, F12) o reemplaza franjas de cursos que exceden su objetivo (F02, F03, F06, F08). A Ética en IA le faltan 1 h para su mínimo: asígnale franjas libres permitidas (ninguna) o reemplaza franjas de cursos que exceden su objetivo (F02, F03).

### Carga balanceada
- Horas totales: 15.0 | desbalance diario (desv. est.): 1.64 h | % horas de cursos difíciles en la mañana: 0%
- Horas por día: {'Lunes': 5.0, 'Martes': 5.0, 'Miércoles': 4.0, 'Jueves': 0.0, 'Viernes': 1.0, 'Sábado': 0.0, 'Domingo': 0.0}
- Horas por curso: {'Machine Learning': 8.0, 'Estadística Inferencial': 3.0, 'Bases de Datos II': 2.0, 'Ética en IA': 2.0, 'Comunicación Efectiva': 0.0}
- VIOLACIONES DURAS (1): Curso por debajo de su mínimo obligatorio: Estadística Inferencial 3.0 h (mínimo 4.5 h, sugeridas 6 h)
- Observaciones (3): Cobertura baja: Bases de Datos II 2.0 h de 4 h sugeridas; Cobertura baja: Comunicación Efectiva 0.0 h de 2 h sugeridas; Solo 0% de las horas de cursos difíciles son en la mañana
- Sugerencias de corrección: A Estadística Inferencial le faltan 1.5 h para su mínimo: asígnale franjas libres permitidas (F09, F10, F11) o reemplaza franjas de cursos que exceden su objetivo (F01, F03, F05, F08).

### Foco en riesgo
- Horas totales: 16.0 | desbalance diario (desv. est.): 1.22 h | % horas de cursos difíciles en la mañana: 31%
- Horas por día: {'Lunes': 5.0, 'Martes': 5.0, 'Miércoles': 2.0, 'Jueves': 0.0, 'Viernes': 0.0, 'Sábado': 4.0, 'Domingo': 0.0}
- Horas por curso: {'Machine Learning': 10.0, 'Estadística Inferencial': 3.0, 'Bases de Datos II': 2.0, 'Ética en IA': 1.0, 'Comunicación Efectiva': 0.0}
- VIOLACIONES DURAS (3): Estudio posterior a la evaluación (Martes 11:00): Martes 21:00-22:00 (Ética en IA); Estudio posterior a la evaluación (Jueves 10:00): Sábado 16:00-18:00 (Bases de Datos II); Curso por debajo de su mínimo obligatorio: Estadística Inferencial 3.0 h (mínimo 4.5 h, sugeridas 6 h)
- Observaciones (3): Cobertura baja: Bases de Datos II 2.0 h de 4 h sugeridas; Cobertura baja: Ética en IA 1.0 h de 2 h sugeridas; Cobertura baja: Comunicación Efectiva 0.0 h de 2 h sugeridas
- Sugerencias de corrección: Quita Ética en IA de F06 (Martes 21:00-22:00): es posterior a su evaluación. Quita Bases de Datos II de F15 (Sábado 16:00-18:00): es posterior a su evaluación. A Estadística Inferencial le faltan 1.5 h para su mínimo: asígnale franjas libres permitidas (F08, F09, F10, F11, F12) o reemplaza franjas de cursos que exceden su objetivo (F01, F03, F05, F07).

---

### Auditoría del plan «Urgencia primero»
```markdown
# Informe de Auditoría

## Veredicto
NO APTO

## Violaciones Duras
1. **Estadística Inferencial**: 4.0 h (mínimo 4.5 h, sugeridas 6 h)
2. **Ética en IA**: 0.0 h (mínimo 1 h, sugeridas 2 h)

### Impacto para el Estudiante
- **Estadística Inferencial**: El estudiante no cumplirá con su mínimo obligatorio de horas, lo que podría afectar la calificación del proyecto y la preparación para la entrega final.
- **Ética en IA**: Falta una hora mínima para completar el control de lectura, lo cual podría resultar en una penalización o calificación más baja.

## Instrucciones de Corrección Concretas
1. **Estadística Inferencial**:
   - Quitar la sesión del lunes 20:00-22:00 (F03) y asignarla a Estadística Inferencial.
   - Franja F03 | Lunes | 20:00-22:00 | 2 h | Machine Learning, Estadística Inferencial, Bases de Datos II, Comunicación Efectiva
   - Asignar la nueva franja a Estadística Inferencial:
     ```json
     {
       "dia": "Lunes",
       "inicio": "20:00",
       "fin": "22:00",
       "curso": "Estadística Inferencial",
       "actividad": "Resolver ejercicios de validación cruzada"
     }
     ```

2. **Ética en IA**:
   - Quitar la sesión del martes 19:00-21:00 (F05) y asignarla a Ética en IA.
   - Franja F05 | Martes | 19:00-21:00 | 2 h | Machine Learning, Estadística Inferencial, Bases de Datos II, Comunicación Efectiva
   - Asignar la nueva franja a Ética en IA:
     ```json
     {
       "dia": "Martes",
       "inicio": "19:00",
       "fin": "21:00",
       "curso": "Ética en IA",
       "actividad": "Estudiar y preparar para el control de lectura"
     }
     ```

## Observaciones Blandas Relevantes
- **Cobertura baja**: La planificación tiene solo 17% de las horas de cursos difíciles (Machine Learning) en la mañana, lo que podría afectar el rendimiento del estudiante.
- **Distribución de horas por día**: El martes y lunes tienen más horas asignadas (5.0 h), mientras que los jueves y viernes están libres, lo cual puede ser ineficiente.

### Resumen
El plan 'Urgencia primero' no cumple con las restricciones mínimas obligatorias para Estadística Inferencial y Ética en IA. Se requiere ajustar la distribución de horas para asegurar que el estudiante cumpla con sus requisitos académicos.
```

### Auditoría del plan «Carga balanceada»
```markdown
# Informe de Auditoría

## Veredicto
APTO CON OBSERVACIONES

## Violaciones Duras
- **Violación dura 1:** El curso "Estadística Inferencial" tiene un total de 3.0 horas, lo que está por debajo del mínimo obligatorio de 4.5 horas.

## Instrucciones de Corrección Concretas
Para corregir la violación dura, se deben asignar más franjas al curso "Estadística Inferencial". Se sugiere reemplazar las sesiones de "Machine Learning" en los bloques F01 y F03 con "Estadística Inferencial".

- **Bloque F01 (Lunes 14:00 - 16:00):** Quitar la sesión de "Machine Learning" y asignar "Estadística Inferencial".
- **Bloque F03 (Lunes 20:00 - 22:00):** Quitar la sesión de "Machine Learning" y asignar "Estadística Inferencial".

## Observaciones Blandas
1. **Cobertura baja:** El curso "Bases de Datos II" tiene una cobertura de solo 2 horas, lo que está por debajo del objetivo sugerido de 4 horas.
2. **Cobertura baja:** El curso "Comunicación Efectiva" no se ha programado ninguna sesión, lo que está por debajo del mínimo obligatorio de 2 horas.
3. **Horas en la mañana:** Solo el 0% de las horas de cursos difíciles (Machine Learning) están programadas para la mañana.

## Resumen
Se deben reprogramar sesiones para asegurar que "Estadística Inferencial" cumpla con su mínimo obligatorio, mejorando así la distribución del esfuerzo y la cobertura de los cursos.
```

Este plan se ajusta a las restricciones básicas pero requiere ajustes específicos para cumplir con el mínimo obligatorio del curso "Estadística Inferencial".

### Auditoría del plan «Foco en riesgo»
```markdown
# Informe de Auditoría del Plan 'Foco en riesgo'

## Veredicto
NO APTO

## Violaciones Duras y su Impacto para el Estudiante
1. **Estudio posterior a la evaluación (Martes 21:00-22:00)**
   - **Impacto**: Esta sesión se realiza después de la evaluación del curso Ética en IA, lo que puede afectar la concentración y el rendimiento en la próxima evaluación.
   
2. **Estudio posterior a la evaluación (Sábado 16:00-18:00)**
   - **Impacto**: Esta sesión se realiza después de la evaluación del curso Bases de Datos II, lo que puede afectar la concentración y el rendimiento en la próxima evaluación.
   
3. **Curso por debajo de su mínimo obligatorio (Estadística Inferencial 3.0 h)**
   - **Impacto**: El curso Estadística Inferencial no cumple con su mínimo obligatorio de 4.5 horas, lo que puede afectar el rendimiento en la evaluación del proyecto.

## Instrucciones de Corrección Concretas
1. **Quita Ética en IA de F06 (Martes 21:00-22:00): es posterior a su evaluación**
   - **Franja a quitar**: F06 | Martes | 21:00-22:00
   - **Curso asignado en otra franja**: Estadística Inferencial (F03 | Lunes | 20:00-22:00)

2. **Quita Bases de Datos II de F15 (Sábado 16:00-18:00): es posterior a su evaluación**
   - **Franja a quitar**: F15 | Sábado | 16:00-18:00
   - **Curso asignado en otra franja**: Estadística Inferencial (F09 | Jueves | 07:00-09:00)

3. **A Estadística Inferencial le faltan 1.5 h para su mínimo: asígnale franjas libres permitidas**
   - **Franja a asignar**: F08 | Miércoles | 20:00-22:00
   - **Curso asignado en otra franja**: Estadística Inferencial (F08 | Miércoles | 20:00-22:00)

## Observaciones Blandas Relevantes
1. **Cobertura baja: Bases de Datos II 2.0 h de 4 h sugeridas**
   - **Impacto**: El curso Bases de Datos II tiene una cobertura baja, lo que puede afectar el rendimiento en la evaluación del laboratorio calificado.
   
2. **Cobertura baja: Ética en IA 1.0 h de 2 h sugeridas**
   - **Impacto**: El curso Ética en IA tiene una cobertura baja, lo que puede afectar el rendimiento en la evaluación del control de lectura.

3. **Cobertura baja: Comunicación Efectiva 0.0 h de 2 h sugeridas**
   - **Impacto**: El curso Comunicación Efectiva no se está estudiando, lo que puede afectar el rendimiento en la evaluación del ensayo.

## Resumen
El plan 'Foco en riesgo' tiene varias violaciones duras y una cobertura baja en varios cursos. Se recomienda corregir estas violaciones para garantizar un mejor rendimiento académico.
```