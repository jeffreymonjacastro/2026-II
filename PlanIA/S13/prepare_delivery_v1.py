"""Prepare a portable delivery copy; never overwrite the working notebooks."""
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import shutil
import nbformat

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "kaggle/v1/outputs_revision1"
DEST = ROOT / "entrega/Hackaton11_v1"
DEST.mkdir(parents=True, exist_ok=True)
notebook = nbformat.read(SOURCE / "hack11_v1_ejecutado.ipynb", 4)
notebook.cells[0].source = notebook.cells[0].source.replace("v1 (kaggle)", "v1 — entrega reproducible")
notebook.cells[0].source += (
    "\n\n**Ejecución portátil:** descomprimir el ZIP, abrir este notebook desde su carpeta "
    "y ejecutar todas las celdas. Los datos están en `data/`; los resultados nuevos "
    "se generan en `salidas/`. `resultados/` conserva la referencia original de Kaggle. "
    "Esta v1 mide F1 de **mal crédito**. No requiere GPU ni credenciales."
)
notebook.cells[1].source = """### Preparación del entorno

Usar Python 3.13 e instalar `requirements.txt`, siguiendo `README.md`.
La copia utiliza CPU y archivos locales; no necesita Kaggle, GPU ni conexión
a Internet después de instalar las dependencias. La semilla, los modelos y
la metodología de v1 se conservan. Las versiones experimentales se fijan a
las de la ejecución original para facilitar la comparación.
"""
config = notebook.cells[4].source
start = config.index('MODE = "kaggle"')
end = config.index('OUT.mkdir(parents=True, exist_ok=True)')
config = config[:start] + '''MODE = "local"
SEED = 42
N_VAL_SIM, N_TEST_SIM = 200, 500
JUNIOR_ACCURACY, SENIOR_ACCURACY = 0.70, 0.95
GROUP = "Equipo por completar"
here = Path.cwd().resolve()
ROOT = next((p for p in [here, *here.parents] if (p / "data/german.data").exists()), None)
if ROOT is None:
    raise FileNotFoundError("Abra el notebook desde la carpeta extraída que contiene data/german.data")
DATA, OUT = ROOT / "data", ROOT / "salidas"
''' + config[end:]
notebook.cells[4].source = config
notebook.cells[36].source = '''ranking_cols = ["Grupo", "Modelo", "F1 - Sólo IA", "F1 - IA + Human", "Mejora", "HACS"]
display(results[ranking_cols].round(4))
results[ranking_cols].to_csv(OUT / "tabla_entrega.csv", index=False)
r = selected_result
st = uncertainty_stats[-1]
xgb_row = results.set_index("Modelo").loc["XGBoost"]
lgbm_row = results.set_index("Modelo").loc["LightGBM"]
'''
discussion = notebook.cells[37].source
for old, new in [
    ("CatBoost fue seleccionado", "{best_name} fue seleccionado"),
    ("51%", "{r['Coverage']:.0%}"),
    ("102 de 200", "{int(r['N_IA'])} de 200"),
    ("Los 98 casos", "Los {int(r['N_Junior'] + r['N_Senior'])} casos"),
    ("0.5946", "{r['F1 - Sólo IA']:.4f}"),
    ("0.8850", "{r['F1 - IA + Human']:.4f}"),
    ("29.04", "{r['Mejora_pp']:.2f}"),
    ("0.7350", "{r['HACS']:.4f}"),
    ("62%", "{xgb_row['Coverage']:.0%}"),
    ("0.7302", "{xgb_row['HACS']:.4f}"),
    ("63.5%", "{lgbm_row['Coverage']:.1%}"),
    ("0.7054", "{lgbm_row['HACS']:.4f}"),
    ("0.8435", "{r['F1_colab_MC_mean']:.4f}"),
    ("el F1 colaborativo medio de CatBoost", "el F1 colaborativo medio de {best_name}"),
    ("más que CatBoost", "más que {best_name}"),
]:
    discussion = discussion.replace(old, new)
notebook.cells[37].source = discussion
# Exclude the platform's notebook from run artifact inventories.
notebook.cells[39].source = notebook.cells[39].source.replace(
    '{"manifest.json", "run_summary.json"}',
    '{"manifest.json", "run_summary.json", "__notebook__.ipynb"}',
)
notebook.metadata.kernelspec = {"display_name":"Python 3.13","language":"python","name":"python3"}
for i,cell in enumerate(notebook.cells):
    cell.id = f"entrega-v1-{i:03d}"
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None
        cell.metadata.pop("execution", None)
        compile(cell.source,cell.id,"exec")
nbformat.validate(notebook)
nbformat.write(notebook, DEST / "hack11_v1_ejecutado.ipynb")
for name in ["german.data", "readme.md", "german.doc"]:
    (DEST / "data").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "data" / name, DEST / "data" / name)
for name in ["tabla_resultados.csv", "tabla_entrega.csv", "estadistica_incertidumbre.csv", "particiones.csv", "run_summary.json"]:
    (DEST / "resultados").mkdir(exist_ok=True)
    shutil.copy2(SOURCE / name, DEST / "resultados" / name)
original = json.loads((SOURCE / "run_summary.json").read_text(encoding="utf-8"))
requirements = dict(original["versions"])
for name in ["joblib", "ipython", "nbformat", "nbclient", "nbconvert", "ipykernel"]:
    requirements[name] = metadata.version(name)
(DEST / "requirements.txt").write_text(
    "# Python 3.13; las librerías experimentales coinciden con Kaggle v1.\n"
    + "\n".join(f"{name}=={version}" for name,version in requirements.items())
    + "\njupyterlab>=4.4,<5\n", encoding="utf-8")
readme = '''# Hackatón 11 — v1: entrega reproducible

Notebook: `hack11_v1_ejecutado.ipynb`. Modelos: XGBoost, LightGBM y CatBoost.
Esta es v1: **positivo = mal crédito** (`y=1`, código original 2).
Completar los nombres del equipo en la portada y en GROUP antes de entregar.

## Ejecutar

Instalar Python 3.13. Descomprimir TODO el ZIP y abrir una terminal en la carpeta
`Hackaton11_v1`, donde están el notebook y `requirements.txt`.

```text
python -m venv .venv
```

Windows (PowerShell), sin necesidad de activar el entorno:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m jupyter lab hack11_v1_ejecutado.ipynb
```

Linux/macOS:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyter lab hack11_v1_ejecutado.ipynb
```

Seleccionar el kernel de ese entorno y usar **Restart Kernel and Run All Cells**.
No ejecutar el notebook dentro del ZIP sin extraerlo. No se requieren GPU,
cuenta de Kaggle, credenciales ni Internet después de instalar dependencias.

Alternativa por terminal (Windows; en Linux/macOS usar `.venv/bin/python`):

```powershell
.venv/Scripts/python.exe -m jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=600 hack11_v1_ejecutado.ipynb
```

## Archivos y resultados

- `data/german.data`: las 1,000 filas originales, 20 atributos y etiqueta.
- `data/readme.md`, `data/german.doc`: diccionario y matriz de costes.
- `resultados/`: referencia original de Kaggle, conservada para comparar.
- `salidas/`: se genera o actualiza al ejecutar, con nuevos CSV, PNG, modelo y resumen.
- `requirements.txt`: versiones de las librerías experimentales de Kaggle y
  herramientas para abrir/ejecutar el notebook.
- `VALIDACION.json`: entorno y comprobaciones de la entrega.

La sección 6.2 muestra los resultados calculados en la ejecución actual; no carga
ni sustituye resultados usando los CSV de referencia. El notebook conserva las
salidas de la ejecución de verificación para que pueda leerse sin ejecutarlo.

Referencia Kaggle (réplica humana 0, prefijada):

| Modelo | F1 IA | F1 colaborativo | HACS |
|---|---:|---:|---:|
| XGBoost | 0.5321 | 0.8037 | 0.7302 |
| LightGBM | 0.5091 | 0.7523 | 0.7054 |
| CatBoost | 0.5946 | 0.8850 | 0.7350 |

Semilla 42; particiones estratificadas 600/200/200; umbral binario 0.5.
Junior 70% y senior 95% son accuracies **simuladas**, no mediciones humanas.
Se guardan 200 simulaciones en validación y 500 en prueba. El F1 principal es
binario de mal crédito y el coste original se calcula como FP + 5 FN.
HACS = 0.6 F1 colaborativo + 0.2 Coverage + 0.2 Efficiency.

Las dependencias experimentales fijadas y las semillas facilitan reproducir los
valores; diferencias de plataforma pueden afectar los últimos decimales. Las
figuras se vuelven a generar. La ejecución remota original usó Python 3.13.16
sobre Linux; `VALIDACION.json` registra la comprobación local de esta copia.

## Fuente

Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository.
https://doi.org/10.24432/C5NC77 — licencia CC BY 4.0.
Guía docente: Sem13_Hackaton11.pdf, proporcionada por el curso.
'''
(DEST / "README.md").write_text(readme,encoding="utf-8")
print("Copia portátil preparada:",DEST)
print("Notebook original SHA256:",hashlib.sha256((SOURCE/"hack11_v1_ejecutado.ipynb").read_bytes()).hexdigest())
