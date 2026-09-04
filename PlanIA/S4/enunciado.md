Nuevas restricciones

- Split 80-20 train-test
- Usar SEED 42
- Muestreo estratificado
- 1000 soles de presupuesto inicial. Ya no se considera que se puede contactar al 20%, en su lugar, inicias con 1000 soles de presupuesto y lo que tienes que determinar es decidir a quiénes intervenir y en qué orden. Luego, se recorre la lista de clientes en orden y, por cada cliente intervenido, se cobran 10 soles. Luego, se revela mediante la etiqueta si la intervención es exitosa o no. Si es exitosa, se añade 100 soles al presupuesto. El objetivo es maximizar el profit final (la ganancia respecto al presupuesto inicial). Considerar que si el presupuesto actual no es suficiente para pagar los 10 soles de intervención, se termina el proeso.
