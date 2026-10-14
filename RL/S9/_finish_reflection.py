import ast
import base64
import json
from pathlib import Path

path = Path(__file__).with_name('Sem9B.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))

def summary(ident):
    c = next(c for c in nb['cells'] if c.get('id') == ident)
    text = ''.join(''.join(o.get('text', [])) for o in c['outputs'] if o['output_type'] == 'stream')
    result = {}
    for line in text.splitlines():
        if '{' in line and 'media_final' in line:
            label, values = line.split('{', 1)
            result[label.strip()] = ast.literal_eval('{' + values)
    assert result, ident
    return result

def table(stats):
    rows = ['| Variante | Baja σ confirmada (ep.) | Buen desempeño estable (ep.) | Media últimos 100 | σ últimos 100 |',
            '|---|---:|---:|---:|---:|']
    for name, s in stats.items():
        low = s['baja_varianza'] or 'No alcanzado'
        conv = s['convergencia'] or 'No alcanzado'
        rows.append(f"| {name} | {low} | {conv} | {s['media_final']:.1f} | {s['std_final']:.1f} |")
    return '\n'.join(rows)

def fastest(stats):
    candidates = {n: s['convergencia'] for n, s in stats.items() if s['convergencia'] is not None}
    if not candidates:
        return 'Ninguna variante alcanzó el criterio de buen desempeño estable dentro del presupuesto.'
    first = min(candidates.values())
    winners = ', '.join(n for n, ep in candidates.items() if ep == first)
    return f'El criterio de buen desempeño estable se confirmó primero para **{winners}**, en el episodio **{first}**.'

q1, q2, q3 = (summary(i) for i in ('t4-q1', 't4-q2', 't4-q3'))
low = min(s['baja_varianza'] for s in q1.values() if s['baja_varianza'] is not None)
low_names = ', '.join(n for n, s in q1.items() if s['baja_varianza'] == low)
quiet1 = min(q1, key=lambda n: q1[n]['std_final'])
quiet2 = min(q2, key=lambda n: q2[n]['std_final'])

answer = f'''## T4 — Preguntas y respuestas

Resultados de la corrida con semilla 42. Cada σ por bloque usa 20 episodios completos. “Baja σ” exige σ ≤ 25 durante tres bloques; “buen desempeño estable” añade media ≥ 475. El episodio indicado es el que confirma el tercer bloque. Las columnas finales usan los últimos 100 episodios del presupuesto indicado. Estas respuestas describen los outputs guardados; si reentrenas, revisa las tablas antes de reutilizar las conclusiones.

### Q1. Grafica la desviación estándar del reward en ventanas de 20 episodios para las 3 variantes. ¿Qué variante converge más rápido a baja varianza?

La gráfica Q1 muestra las tres curvas y su recompensa media para los 1000 episodios de T1–T3.

{table(q1)}

El primer cumplimiento de baja σ corresponde a **{low_names}**, en el episodio **{low}**. Eso puede ocurrir con reward bajo: por sí solo no demuestra aprendizaje. {fastest(q1)} La menor σ de los últimos 100 episodios corresponde a **{quiet1}** ({q1[quiet1]['std_final']:.1f} puntos); la media de esa misma ventana es {q1[quiet1]['media_final']:.1f}.

Un baseline puede reducir la variación del estimador del gradiente al restar una referencia; un V(s) bien aprendido compara cada retorno con lo esperable desde ese estado. La normalización por episodio también cambia la escala de la actualización. Ninguna de estas ideas garantiza que una variante gane en toda corrida. Además, aquí medimos dispersión del **reward entre episodios**, no directamente la varianza del gradiente.

### Q2. Compara tus curvas de REINFORCE con las de DQN (Lab 7B) en CartPole. ¿Cuál converge más rápido? ¿Cuál tiene menos varianza? ¿Por qué?

La gráfica Q2 y esta tabla usan el mismo presupuesto de **500 episodios**, con reward total sin descuento para todos los algoritmos.

{table(q2)}

{fastest(q2)} La menor desviación estándar de los últimos 100 episodios corresponde a **{quiet2}** ({q2[quiet2]['std_final']:.1f} puntos), con media {q2[quiet2]['media_final']:.1f}. Una curva constante con reward bajo también puede tener σ pequeña, por eso esta comparación incluye la media.

DQN reutiliza transiciones con replay buffer y aleatoriza los minibatches, reduciendo correlaciones entre muestras consecutivas; la target network estabiliza sus objetivos. REINFORCE usa retornos Monte Carlo de episodios completos, que pueden producir actualizaciones más variables. DQN hace bootstrap y puede introducir errores de estimación. Monte Carlo evita ese bootstrap; la afirmación “sin sesgo” requiere matices cuando usamos una media y normalización calculadas con el propio episodio. La rapidez observada también depende de hiperparámetros y exploración. DQN actualiza por paso, REINFORCE por episodio: esta comparación no demuestra cuál usa menos tiempo o actualizaciones. Una sola semilla no establece un ganador general.

### Q3. ¿Qué pasaría si aumentas gamma de 0.99 a 1.0? ¿Los retornos Gₜ crecen o se mantienen? ¿Afecta la convergencia?

Con **la misma trayectoria de CartPole**, los retornos crecen o se mantienen: las recompensas futuras +1 ya no se descuentan. En un episodio óptimo de 500 pasos, G₀ pasa de **99.34** a **500**; en el último paso ambos retornos valen 1. La primera gráfica Q3 comprueba ese efecto sin mezclarlo con cambios de política. La recompensa total del episodio sigue siendo 500 en ambos casos.

El reentrenamiento con baseline batch, normalización en ambos casos y 500 episodios produjo:

{table(q3)}

{fastest(q3)} Cambiar gamma modifica cuánto pesan las consecuencias lejanas y puede cambiar la convergencia, pero no garantiza acelerarla ni frenarla. En trayectorias de distintas duraciones, gamma=1 produce retornos de mayor escala y puede ampliar su dispersión; eso puede hacer más importante normalizar. No significa que todo retorno mayor sea automáticamente más ruidoso. Después de reentrenar, las trayectorias cambian, así que G₀ no tiene por qué aumentar en todos los episodios emparejados. CartPole tiene un límite de 500 pasos, por lo que gamma=1 sigue dando retornos finitos.

Fuentes conceptuales: [Spinning Up — policy gradients y baselines](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) y [Mnih et al. — DQN, replay y target network](https://www.nature.com/articles/nature14236). Las mediciones proceden de las celdas de este notebook y de las definiciones del Lab 7B.
'''
nb['cells'][-1]['source'] = answer.splitlines(True)
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
for c in nb['cells']:
    if c.get('id') in {'t4-q1', 't4-q2', 't4-q3'}:
        image_index = 0
        for out in c['outputs']:
            data = out.get('data', {}).get('image/png')
            if data:
                if isinstance(data, list):
                    data = ''.join(data)
                target = Path(__file__).parent / f"_preview_{c['id']}_{image_index}.png"
                target.write_bytes(base64.b64decode(data))
                image_index += 1
print(answer)
