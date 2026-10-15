# Guía  de Implementación: Advantage Actor-Critic (A2C)
**Curso:** Aprendizaje por Refuerzo  
**Laboratorio:** S10B - Implementación de A2C en PyTorch  
**Entorno:** `CartPole-v1`

---

## 1. Fundamento Teórico: De REINFORCE a Actor-Critic (A2C)

En algoritmos basados exclusivamente en gradiente de política como **REINFORCE**, la actualización de la red depende del retorno acumulado Monte Carlo $G_t$:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

Aunque este estimador no tiene sesgo, sufre de **alta varianza**: pequeñas diferencias en trayectorias futuras pueden cambiar drásticamente el valor de $G_t$, haciendo el aprendizaje lento e inestable.

### La Función de Ventaja (Advantage)
Para resolver este problema, sustituimos el retorno crudo $G_t$ por la **Función de Ventaja** $A(s, a)$:

$$A(s, a) = Q(s, a) - V(s)$$

- **$Q(s, a)$**: Qué tan buena es una acción específica $a$ en el estado $s$.
- **$V(s)$**: Qué tan bueno es en promedio estar en el estado $s$ bajo la política actual.
- **$A(s, a)$**: Mide si la acción $a$ fue **mejor ($A > 0$) o peor ($A < 0$) que el promedio esperado**.

En la arquitectura **Actor-Critic**:
- **El Actor ($\pi_\theta$)**: Aprende qué acción tomar (política).
- **El Crítico ($V_\phi$)**: Aprende a predecir el valor promedio de los estados $V(s)$, sirviendo como línea base dinámica (baseline) que reduce drásticamente la varianza.

---

## 2. Anatomía de la Red Neuronal Compartida (`ActorCriticNet`)

En lugar de crear dos redes neuronales independientes para el actor y el crítico, se utiliza una **red con tronco compartido (backbone) y dos cabezas**.

```python
class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=256):
        super().__init__()
        # 1. Tronco compartido: extractor de características
        self.tronco = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),    nn.Tanh(),
        )
        # 2. Cabeza del Actor: produce logits sobre acciones discretas
        self.cabeza_actor   = nn.Linear(hidden, n_actions)
        
        # 3. Cabeza del Crítico: estima el valor escalar V(s)
        self.cabeza_critico = nn.Linear(hidden, 1)

    def forward(self, x):
        feat  = self.tronco(x)
        dist  = Categorical(logits=self.cabeza_actor(feat))
        value = self.cabeza_critico(feat).squeeze(-1)
        return dist, value
```

### Claves de Diseño:
1. **Tronco Compartido:** Aprende representaciones latentes comunes sobre la física del entorno que benefician tanto a la evaluación del estado como a la toma de decisiones, ahorrando parámetros y cómputo.
2. **Activación `Tanh` vs `ReLU`:**  
   En gradientes de política, `ReLU` puede provocar el problema de **neuronas muertas** si las activaciones caen permanentemente por debajo de 0 tras una mala actualización. `Tanh` proporciona gradientes suaves, continuos y acotados entre $[-1, 1]$, ofreciendo mayor estabilidad numérica.
3. **Distribución Categórica (`Categorical`):**  
   Recibe los *logits* no normalizados directamente de la cabeza del actor para construir numéricamente una distribución de probabilidad mediante softmax interno estable.
4. **`squeeze(-1)` en el Crítico:**  
   La cabeza del crítico produce una salida de dimensión `(batch, 1)`. Con `squeeze(-1)` se transforma en un escalar o vector unidimensional `(batch,)` para alinearse con los tensores de retornos y recompensas.

---

## 3. El Ciclo de Interacción y Entrenamiento (`train_a2c_episode`)

El entrenamiento recolecta una trayectoria y luego optimiza los pesos mediante retropropagación de la función de pérdida combinada de 3 términos.

### 3.1. Recolección de Experiencia
```python
while not done:
    s    = torch.FloatTensor(obs)
    dist, val = model(s)
    action    = dist.sample()                    # Muestreo estocástico (exploración)
    obs, r, term, trunc, _ = env.step(action.item())
    done = term or trunc
    
    log_probs.append(dist.log_prob(action))      # log π(a_t | s_t)
    values.append(val)                           # V(s_t) predicho por el crítico
    rewards.append(r)                            # Recompensa inmediata r_{t+1}
    entropies.append(dist.entropy())             # Entropía H(π(·|s_t))
```

- **Muestreo con `dist.sample()`:** Permite que el agente mantenga exploración activa según las probabilidades actuales.
- **`dist.log_prob(action)`:** Registra el logaritmo de probabilidad diferenciable necesario para calcular el gradiente de política.
- **`dist.entropy()`:** Mide el grado de incertidumbre/dispersión de la política en ese instante.

---

### 3.2. Cálculo del Retorno y Estimación de Ventaja

```python
# Retornos descontados acumulados G_t (recorrido inverso)
G, returns = 0.0, []
for r in reversed(rewards):
    G = r + gamma * G
    returns.insert(0, G)

returns = torch.tensor(returns, dtype=torch.float32)
# Estandarización de retornos
returns = (returns - returns.mean()) / (returns.std() + 1e-8)

log_probs  = torch.stack(log_probs)
values     = torch.stack(values)
entropies  = torch.stack(entropies)

# Ventaja: A_t = G_t - V(s_t)
advantages = returns - values.detach()
```

#### ¿Por qué es crucial el `.detach()` en `values`?
La ventaja $A_t$ cumple el rol de **señal de refuerzo externa (ponderación escalar)** que le indica al actor en qué dirección y con qué magnitud actualizar sus preferencias.  
Si **no** colocamos `.detach()`, el gradiente del actor intentaría fluir hacia atrás a través de $V(s_t)$, forzando al crítico a cambiar sus predicciones no para evaluar mejor el estado, sino para minimizar artificialmente la pérdida del actor. El `.detach()` corta el grafo computacional en esa bifurcación.

#### Normalización de Retornos:
La normalización restringe los retornos a media 0 y varianza unitaria. Esto evita gradientes desmesurados cuando las trayectorias son muy largas (como en CartPole con recompensas de 500 puntos).

---

### 3.3. La Función de Pérdida Total (3 Términos)

La función objetivo unificada de A2C equilibra tres propósitos:

$$\mathcal{L}_{total} = \mathcal{L}_{actor} + c_v \cdot \mathcal{L}_{critic} - c_e \cdot \mathcal{H}(\pi_\theta)$$

En código:
```python
# 1. Pérdida del Actor (Policy Gradient ponderado por ventaja)
L_actor   = -(log_probs * advantages).mean()

# 2. Pérdida del Crítico (Error Cuadrático Medio / MSE)
L_critic  = c_v * (returns - values).pow(2).mean()

# 3. Bonificación por Entropía (evita colapso prematuro)
L_entropy = -c_e * entropies.mean()

# Pérdida total
L_total   = L_actor + L_critic + L_entropy
```

#### Análisis detallado de los 3 términos:
1. **$\mathcal{L}_{actor} = -\mathbb{E}[A_t \cdot \log \pi_\theta(a_t|s_t)]$:**  
   - Si la acción fue mejor que el promedio ($A_t > 0$), el signo negativo hace que minimizar la pérdida incremente la probabilidad de volver a elegir $a_t$.  
   - Si fue peor ($A_t < 0$), disminuye su probabilidad futura.
2. **$c_v \cdot \mathcal{L}_{critic} = c_v \cdot \mathbb{E}[(G_t - V_\phi(s_t))^2]$:**  
   - Entrena la cabeza del crítico mediante regresión cuadrática para ajustar sus predicciones de valor al retorno observado.  
   - $c_v = 0.5$ modera el impacto del crítico para que no domine el gradiente del tronco compartido.
3. **$-c_e \cdot \mathcal{H}(\pi_\theta)$:**  
   - La entropía $\mathcal{H}(\pi) = -\sum \pi(a|s) \log \pi(a|s)$ es máxima cuando la política asigna probabilidades uniformes (máxima exploración) y mínima cuando es determinista.  
   - Al restar $c_e \cdot \mathcal{H}$ a la función de costo, el optimizador busca **maximizar la entropía**, evitando que el agente se cierre prematuramente a una sola acción antes de descubrir estrategias óptimas.

---

### 3.4. Paso de Optimización y Gradient Clipping

```python
optimizer.zero_grad()
L_total.backward()
# Gradient Clipping para estabilidad numérica
nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
optimizer.step()
```

- **Gradient Clipping (`max_grad_norm = 0.5`):**  
  En métodos de gradiente de política, un solo paso con un gradiente excesivamente grande puede deteriorar radicalmente los pesos y destruir una buena política de forma irreversible. El recorte garantiza que la norma euclidiana del vector de gradientes nunca supere $0.5$.

---

## 4. Diagnóstico y Monitoreo de Métricas

Para comprobar que el entrenamiento es saludable, se analizan 4 curvas de forma simultánea:

| Métrica | Comportamiento Esperado | Interpretación Diagnóstica |
| :--- | :--- | :--- |
| **Recompensa (Reward)** | Debe ascender hasta el techo del entorno (500 en CartPole-v1). | Mide la efectividad real del agente. Una curva con media móvil ascendente indica convergencia. |
| **Pérdida del Actor ($\mathcal{L}_{actor}$)** | Debe fluctuar y mantenerse cerca de 0. | A medida que el crítico aprende, la ventaja $A_t \approx 0$ (no hay sorpresas); por ende, $\mathcal{L}_{actor}$ se estabiliza. No debe explotar. |
| **Pérdida del Crítico ($\mathcal{L}_{critic}$)** | Comienza alta y disminuye progresivamente. | Confirma que el crítico está aprendiendo a aproximar con precisión el valor real de los estados. |
| **Entropía ($\mathcal{H}(\pi)$)** | Inicia cerca de su valor máximo ($\ln(2) \approx 0.693$) y desciende de forma gradual y controlada. | Refleja que el agente va ganando certeza. Si cae a 0 bruscamente en los primeros episodios, la política ha colapsado prematuramente. |

---

## 5. Experimentos y Comparaciones

### 5.1. A2C vs REINFORCE
- **REINFORCE (sin baseline):** Utiliza retornos Monte Carlo directamente. Tiende a presentar trayectorias oscilantes, mayor dispersión y tardanza en alcanzar el máximo de 500 puntos.
- **A2C:** Gracias a la sustracción de la función de valor estimada por el crítico ($A_t = G_t - V(s_t)$), la señal de aprendizaje tiene una varianza sustancialmente menor, lo que se traduce en un aprendizaje más rápido, suave y confiable.

### 5.2. Ablación del Coeficiente de Entropía ($c_e$)
El parámetro $c_e$ modula el balance exploración-explotación:
- **$c_e = 0.0$ (Sin entropía):** Riesgo elevado de colapso prematuro. El agente puede aprender una política determinista subóptima y estancarse.
- **$c_e = 0.01$ (Valor estándar):** Proporciona la exploración necesaria en las fases iniciales y permite converger de forma sólida a la política óptima.
- **$c_e = 0.05$ (Demasiada entropía):** Obliga a la política a ser demasiado estocástica, dificultando que el agente explote el conocimiento adquirido y estabilice el equilibrio del péndulo.

---

## 6. Resumen de Hiperparámetros Recomendados

```python
hiperparametros = {
    "entorno": "CartPole-v1",
    "hidden_dim": 256,
    "activacion": "Tanh",
    "lr": 1e-3,
    "gamma": 0.99,
    "c_v": 0.5,        # Ponderación del crítico en la loss total
    "c_e": 0.01,       # Regularizador de entropía
    "max_grad_norm": 0.5, # Umbral de clipping de gradiente
    "n_episodios": 2000
}
```
