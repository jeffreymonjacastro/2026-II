import json
from pathlib import Path

path = Path(__file__).with_name('Sem9B.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))

def cell(kind, source, ident):
    result = dict(cell_type=kind, id=ident, metadata={}, source=source.strip().splitlines(True))
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result

nb['cells'].insert(1, cell('code', '''
# Semilla reproducible para los entrenamientos originales (T1–T3).
import torch
import numpy as np
torch.set_num_threads(1)
torch.manual_seed(42)
np.random.seed(42)
''', 't4-seed'))
# Seed each variant independently; leave the original logic and logging intact.
for c in nb['cells']:
    s = ''.join(c['source'])
    if 'env = gym.make("CartPole-v1")' in s:
        s = s.replace('policy = PolicyNet(4, 2)', 'env.reset(seed=42)\npolicy = PolicyNet(4, 2)', 1)
    elif '# Reinicializar politica' in s:
        s = s.replace('# Reinicializar politica', '# Reinicializar politica\ntorch.manual_seed(42)\nenv.reset(seed=42)')
    elif 'policy_v = PolicyNet' in s:
        s = s.replace('policy_v = PolicyNet', 'torch.manual_seed(42)\nenv.reset(seed=42)\npolicy_v = PolicyNet', 1)
    c['source'] = s.splitlines(True)

nb['cells'].extend([
cell('markdown', '''
## T4 — Análisis de varianza y comparación con DQN

Ejecuta T1–T3 antes de estas celdas. Sus listas `rewards_*` contienen **G₀ descontado**, no el reward total. En CartPole cada paso da +1: recuperamos la duración del episodio a partir de G₀ para comparar con DQN en puntos (máximo 500). La conversión solo es válida para estos entrenamientos con gamma=0.99 y recompensa +1.

Usamos bloques completos, no solapados, de 20 episodios y `np.std(ddof=0)`. Definimos baja variación como σ ≤ 25 y buen desempeño como media ≥ 475, durante tres bloques seguidos. Son criterios operativos, no una prueba estadística de convergencia. Una sola semilla permite describir esta corrida; para generalizar hay que repetir con varias semillas.
''', 't4-intro'),
cell('code', '''
# Q1: desviación estándar y desempeño de las tres variantes.
from IPython.display import display, Markdown

def reward_cartpole(g0, gamma=0.99):
    g0 = np.asarray(g0, dtype=float)
    if gamma == 1:
        return g0.copy()
    durations = np.rint(np.log1p(-(1 - gamma) * g0) / np.log(gamma))
    assert np.all((durations >= 1) & (durations <= 500))
    assert np.allclose((1 - gamma ** durations) / (1 - gamma), g0, atol=1e-4)
    return durations

rewards_t4 = {
    "Sin baseline": reward_cartpole(rewards_no_baseline),
    "Baseline batch": reward_cartpole(rewards_baseline_batch),
    "Baseline V(s)": reward_cartpole(rewards_value),
}
WINDOW, STD_MAX, MEAN_MIN, STREAK = 20, 25, 475, 3
colors_t4 = {"Sin baseline": "#EA580C", "Baseline batch": "#008568",
             "Baseline V(s)": "#007CB0", "DQN": "#7040A0"}

def block_stats(rewards):
    r = np.asarray(rewards, dtype=float)
    blocks = r[:len(r) // WINDOW * WINDOW].reshape(-1, WINDOW)
    return np.arange(1, len(blocks) + 1) * WINDOW, blocks.mean(1), blocks.std(1)

def first_streak(episodes, condition):
    for i in range(len(condition) - STREAK + 1):
        if np.all(condition[i:i + STREAK]):
            return int(episodes[i + STREAK - 1])  # episodio que confirma 3 bloques
    return None

def summarize(rewards):
    episodes, means, stds = block_stats(rewards)
    return {
        "baja_varianza": first_streak(episodes, stds <= STD_MAX),
        "convergencia": first_streak(episodes, (stds <= STD_MAX) & (means >= MEAN_MIN)),
        "media_final": float(np.mean(rewards[-100:])),
        "std_final": float(np.std(rewards[-100:])),
    }

assert first_streak(np.array([20, 40, 60, 80]), np.array([False, True, True, True])) == 80
assert np.array_equal(reward_cartpole((1 - 0.99 ** np.array([1, 20, 500])) / 0.01), [1, 20, 500])

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for name, rewards in rewards_t4.items():
    episodes, means, stds = block_stats(rewards)
    axes[0].plot(episodes, stds, label=name, color=colors_t4[name])
    axes[1].plot(episodes, means, label=name, color=colors_t4[name])
axes[0].axhline(STD_MAX, color="0.3", linestyle="--", label="σ ≤ 25")
axes[1].axhline(MEAN_MIN, color="0.3", linestyle="--", label="Media ≥ 475")
axes[0].set(title="Q1 — Variación del reward en bloques de 20 episodios", ylabel="Desviación estándar (puntos)")
axes[1].set(xlabel="Episodio final del bloque", ylabel="Reward medio (puntos)", ylim=(0, 520))
for ax in axes:
    ax.grid(alpha=0.2)
    ax.legend()
plt.tight_layout()
plt.show()
summary_q1 = {name: summarize(r) for name, r in rewards_t4.items()}
for name, stats in summary_q1.items():
    print(name, stats)
''', 't4-q1'),
cell('markdown', '''
### Q2 — REINFORCE frente al DQN del Lab 7B

Reutilizamos las clases y la función `train` de `RL/S7/Sem 7B Lab.ipynb`, sin ejecutar sus experimentos ni modificar ese archivo. El presupuesto común es de 500 episodios; REINFORCE se recorta a sus primeros 500. Medimos rapidez por episodios, no por tiempo ni número de actualizaciones: DQN actualiza por paso y REINFORCE por episodio. Las curvas son de entrenamiento, con exploración distinta en cada algoritmo.
''', 't4-q2-intro'),
cell('code', '''
import ast
import json
import random
from pathlib import Path

# Localizar la fuente tanto desde S9 como desde la raíz del repositorio.
dqn_path = next((p for p in [Path("../S7/Sem 7B Lab.ipynb"),
                            Path("RL/S7/Sem 7B Lab.ipynb")]
                 if p.is_file()), None)
if dqn_path is None:
    raise FileNotFoundError("No se encontró RL/S7/Sem 7B Lab.ipynb")
dqn_nb = json.loads(dqn_path.read_text(encoding="utf-8"))
dqn_scope = {}
definitions = []
for c in dqn_nb["cells"]:
    if c["cell_type"] != "code":
        continue
    tree = ast.parse("".join(c["source"]))
    definitions.extend(node for node in tree.body
                       if isinstance(node, (ast.Import, ast.ImportFrom))
                       or isinstance(node, (ast.ClassDef, ast.FunctionDef))
                       and node.name in {"QNetwork", "ReplayBuffer", "DQNAgent", "train"})
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(dqn_path), "exec"), dqn_scope)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
env_dqn_t4 = gym.make("CartPole-v1")
try:
    agent_dqn_t4 = dqn_scope["DQNAgent"](state_dim=4, n_actions=2)
    rewards_dqn_t4, _ = dqn_scope["train"](agent_dqn_t4, env_dqn_t4, n_episodes=500)
finally:
    env_dqn_t4.close()

comparison_t4 = {name: r[:500] for name, r in rewards_t4.items()}
comparison_t4["DQN"] = np.array(rewards_dqn_t4)
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for name, rewards in comparison_t4.items():
    episodes, means, stds = block_stats(rewards)
    axes[0].plot(episodes, means, color=colors_t4[name], label=name)
    axes[1].plot(episodes, stds, color=colors_t4[name], label=name)
axes[0].axhline(MEAN_MIN, color="0.3", linestyle="--", label="Media ≥ 475")
axes[1].axhline(STD_MAX, color="0.3", linestyle="--", label="σ ≤ 25")
axes[0].set(title="Q2 — REINFORCE vs DQN: primeros 500 episodios", ylabel="Reward medio (puntos)", ylim=(0, 520))
axes[1].set(xlabel="Episodio final del bloque", ylabel="Desviación estándar (puntos)")
for ax in axes:
    ax.grid(alpha=0.2)
    ax.legend()
plt.tight_layout()
plt.show()
summary_q2 = {name: summarize(r) for name, r in comparison_t4.items()}
for name, stats in summary_q2.items():
    print(name, stats)
''', 't4-q2'),
cell('markdown', '''
### Q3 — Cambiar gamma de 0.99 a 1.0

Primero aislamos el efecto sobre Gₜ usando **la misma trayectoria óptima** de 500 recompensas +1. Después reentrenamos REINFORCE con baseline batch para ambos gamma: misma semilla, arquitectura, learning rate y presupuesto (500 episodios). La normalización se mantiene en ambos casos. Las trayectorias pueden divergir al cambiar la política; esta prueba no aísla el efecto de quitar la normalización.
''', 't4-q3-intro'),
cell('code', '''
def discounted_returns(rewards, gamma):
    returns = np.empty(len(rewards), dtype=float)
    G = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns

optimal_returns = {gamma: discounted_returns(np.ones(500), gamma) for gamma in (0.99, 1.0)}
assert np.all(optimal_returns[1.0] >= optimal_returns[0.99])
assert optimal_returns[1.0][0] == 500
fig, ax = plt.subplots(figsize=(10, 4))
for gamma, returns in optimal_returns.items():
    ax.plot(np.arange(500), returns, label=f"gamma={gamma}; G₀={returns[0]:.2f}")
ax.set(title="Q3 — Gₜ sobre la misma trayectoria de 500 pasos", xlabel="Paso t", ylabel="Retorno Gₜ")
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()

def train_gamma(gamma, episodes=500, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    local_env = gym.make("CartPole-v1")
    local_policy = PolicyNet(4, 2)
    local_opt = torch.optim.Adam(local_policy.parameters(), lr=1e-3)
    rewards_hist, g0_hist = [], []
    try:
        for ep in range(episodes):
            obs, _ = local_env.reset(seed=seed + ep)
            log_probs, rewards = [], []
            while True:
                dist = local_policy(torch.as_tensor(obs, dtype=torch.float32))
                action = dist.sample()
                obs, r, done, trunc, _ = local_env.step(action.item())
                log_probs.append(dist.log_prob(action))
                rewards.append(r)
                if done or trunc:
                    break
            returns = torch.tensor(discounted_returns(rewards, gamma), dtype=torch.float32)
            normalized = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
            loss = -(torch.stack(log_probs) * normalized).mean()
            local_opt.zero_grad()
            loss.backward()
            local_opt.step()
            rewards_hist.append(sum(rewards))
            g0_hist.append(returns[0].item())
    finally:
        local_env.close()
    return np.array(rewards_hist), np.array(g0_hist)

gamma_runs = {}
for gamma in (0.99, 1.0):
    print(f"Entrenando baseline batch con gamma={gamma}...")
    gamma_runs[gamma] = train_gamma(gamma)
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for gamma, (rewards, g0) in gamma_runs.items():
    episodes, means, stds = block_stats(rewards)
    axes[0].plot(episodes, means, label=f"gamma={gamma}")
    axes[1].plot(episodes, stds, label=f"gamma={gamma}")
axes[0].axhline(MEAN_MIN, color="0.3", linestyle="--", label="Media ≥ 475")
axes[1].axhline(STD_MAX, color="0.3", linestyle="--", label="σ ≤ 25")
axes[0].set(title="Q3 — Efecto de gamma con normalización en ambos entrenamientos", ylabel="Reward medio (puntos)", ylim=(0, 520))
axes[1].set(xlabel="Episodio final del bloque", ylabel="Desviación estándar (puntos)")
for ax in axes:
    ax.legend()
    ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()
summary_q3 = {gamma: summarize(run[0]) for gamma, run in gamma_runs.items()}
for gamma, stats in summary_q3.items():
    print(f"gamma={gamma}", stats)
''', 't4-q3'),
cell('markdown', '''
## T4 — Preguntas y respuestas

Las respuestas numéricas se completan con los resultados ejecutados de las celdas anteriores.
''', 't4-answers')
])
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
