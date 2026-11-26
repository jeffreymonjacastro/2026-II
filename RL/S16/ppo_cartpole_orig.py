"""
DS5345 — Aprendizaje por Refuerzo (UTEC 2026-II)
Examen Final — Sección II: PPO-Clip + GAE en CartPole-v1

Instrucciones:
Complete únicamente los bloques marcados con TODO 1, TODO 2, TODO 3 y TODO 4.
No modifique las firmas de las funciones ni el entorno de evaluación.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt


# ==============================================================================
# Hiperparámetros (calibrados para converger rápido en CPU)
# ==============================================================================
ENV_NAME = "CartPole-v1"
SEED = 42
TOTAL_UPDATES = 60         # Número de iteraciones de recolección + actualización
ROLLOUT_STEPS = 1024       # Pasos por rollout
GAMMA = 0.99               # Factor de descuento
GAE_LAMBDA = 0.95          # Parámetro lambda de GAE
CLIP_EPS = 0.2             # Épsilon de recorte de PPO
PPO_EPOCHS = 4             # Épocas de reutilización del batch
MINI_BATCH_SIZE = 64       # Tamaño de mini-batch
LR = 5e-4                  # Tasa de aprendizaje (Adam)
C_VALUE = 0.5              # Coeficiente de la pérdida del crítico (c_v)
C_ENTROPY = 0.01           # Coeficiente de regularización por entropía (c_e)
MAX_GRAD_NORM = 0.5        # Recorte de norma de gradiente


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==============================================================================
# Red Actor-Crítico con redes separadas (provista)
# ==============================================================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


# ==============================================================================
# Recolección de trayectorias (provista)
# ==============================================================================
def collect_rollout(env, model, state, rollout_steps, ep_reward, episode_rewards):
    states = []
    actions = []
    old_log_probs = []
    rewards = []
    dones = []
    values = []

    for _ in range(rollout_steps):
        state_t = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_t, log_prob_t, _, value_t = model.get_action_and_value(state_t)

        next_state, reward, terminated, truncated, _ = env.step(action_t.item())
        done = terminated or truncated

        states.append(state_t)
        actions.append(action_t)
        old_log_probs.append(log_prob_t)
        rewards.append(reward)
        dones.append(float(done))
        values.append(value_t)

        ep_reward += reward
        if done:
            episode_rewards.append(ep_reward)
            ep_reward = 0.0
            state, _ = env.reset()
        else:
            state = next_state

    with torch.no_grad():
        _, next_value = model.forward(torch.tensor(state, dtype=torch.float32))

    rollout = {
        "states": torch.stack(states),
        "actions": torch.stack(actions),
        "old_log_probs": torch.stack(old_log_probs),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "dones": torch.tensor(dones, dtype=torch.float32),
        "values": torch.stack(values),
        "next_value": next_value,
    }
    return rollout, state, ep_reward


# ==============================================================================
# TODO 1: Cálculo de GAE(lambda) y Retornos Objetivo
# ==============================================================================
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = GAMMA,
    lam: float = GAE_LAMBDA,
):
    """
    Calcula las ventajas generalizadas (GAE) y los retornos objetivo R_t = A_t + V(s_t).

    Pasos sugeridos:
    1. Inicializar `advantages = torch.zeros_like(rewards)` y `gae = 0.0`.
    2. Recorrer los pasos t desde T-1 hasta 0:
       - next_val = next_value si t == T-1, sino values[t + 1]
       - mask = 1.0 - dones[t]   (corta el bootstrapping en estados terminales)
       - delta = rewards[t] + gamma * next_val * mask - values[t]
       - gae = delta + gamma * lam * mask * gae
       - advantages[t] = gae
    3. Calcular `returns = advantages + values` ANTES de normalizar.
    4. Normalizar `advantages` restando su media y dividiendo por (std + 1e-8).

    Retorna:
        advantages (Tensor[T]), returns (Tensor[T])
    """
    # --- INICIO TODO 1 ---
    raise NotImplementedError("Complete TODO 1: compute_gae")
    # --- FIN TODO 1 ---


# ==============================================================================
# TODO 2: Ratio de Probabilidad y Pérdida Recortada (PPO-Clip)
# ==============================================================================
def compute_ppo_clip_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = CLIP_EPS,
):
    """
    Calcula la pérdida del actor L^{CLIP} (con signo negativo para minimización)
    y la fracción de muestras cuyo ratio fue recortado (clip_fraction).

    Fórmulas:
        ratio = exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -mean(min(surr1, surr2))
        clip_fraction = mean(|ratio - 1| > clip_eps)

    Retorna:
        policy_loss (Tensor escalar), clip_fraction (float)
    """
    # --- INICIO TODO 2 ---
    raise NotImplementedError("Complete TODO 2: compute_ppo_clip_loss")
    # --- FIN TODO 2 ---


# ==============================================================================
# TODO 3: Actualización PPO por K épocas en mini-batches
# ==============================================================================
def ppo_update(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    rollout: dict,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    ppo_epochs: int = PPO_EPOCHS,
    mini_batch_size: int = MINI_BATCH_SIZE,
    clip_eps: float = CLIP_EPS,
    c_value: float = C_VALUE,
    c_entropy: float = C_ENTROPY,
    max_grad_norm: float = MAX_GRAD_NORM,
) -> float:
    """
    Ejecuta K épocas de actualización PPO usando mini-batches aleatorios.

    Pasos sugeridos:
    1. Asegurar que `old_log_probs`, `advantages` y `returns` estén desconectados
       del grafo computacional usando `.detach()`.
    2. En cada época (0 .. ppo_epochs - 1):
       - Permutar los índices del batch con `torch.randperm(batch_size)`.
       - Iterar sobre los mini-batches de tamaño `mini_batch_size`:
         a) Re-evaluar acciones con `model.get_action_and_value(mb_states, mb_actions)`.
         b) Calcular `policy_loss` y `clip_frac` llamando a `compute_ppo_clip_loss(...)`.
         c) Calcular `value_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()`.
         d) Calcular `entropy_bonus = entropy.mean()`.
         e) Combinar `total_loss = policy_loss + c_value * value_loss - c_entropy * entropy_bonus`.
         f) Hacer `optimizer.zero_grad()`, `total_loss.backward()`,
            `nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)` y `optimizer.step()`.
         g) Acumular `clip_frac` en una lista.
    3. Retornar el promedio de `clip_frac` (float) de toda la actualización.
    """
    # --- INICIO TODO 3 ---
    raise NotImplementedError("Complete TODO 3: ppo_update")
    # --- FIN TODO 3 ---


# ==============================================================================
# TODO 4: Análisis Visual
# ==============================================================================
def plot_results(episode_rewards: list, clip_fractions: list) -> None:
    """
    Genera una figura con dos subgráficos:
    1. Recompensa por episodio junto con su media móvil (ventana de 20 episodios).
    2. Evolución de la fracción de clipping (clip_fraction) por actualización de PPO.

    Guarde la gráfica como 'ppo_cartpole_resultados.png' y muéstrela con plt.show().
    """
    # --- INICIO TODO 4 ---
    raise NotImplementedError("Complete TODO 4: plot_results")
    # --- FIN TODO 4 ---


# ==============================================================================
# Bucle principal de entrenamiento (provisto)
# ==============================================================================
def main():
    set_seed(SEED)
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = ActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    state, _ = env.reset(seed=SEED)
    ep_reward = 0.0
    episode_rewards = []
    clip_fractions = []

    for update in range(1, TOTAL_UPDATES + 1):
        rollout, state, ep_reward = collect_rollout(
            env, model, state, ROLLOUT_STEPS, ep_reward, episode_rewards
        )

        advantages, returns = compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
            rollout["next_value"],
            gamma=GAMMA,
            lam=GAE_LAMBDA,
        )

        mean_clip = ppo_update(model, optimizer, rollout, advantages, returns)
        clip_fractions.append(mean_clip)

        if update % 10 == 0 and len(episode_rewards) > 0:
            recent_avg = np.mean(episode_rewards[-20:])
            print(
                f"Update {update:3d}/{TOTAL_UPDATES} | "
                f"Episodios: {len(episode_rewards):4d} | "
                f"Recompensa media (ult. 20): {recent_avg:6.1f} | "
                f"Clip frac: {mean_clip:.3f}"
            )

    env.close()
    plot_results(episode_rewards, clip_fractions)


if __name__ == "__main__":
    main()
