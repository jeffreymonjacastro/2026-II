"""Execute an extracted delivery copy, verify v1 results, then create the final ZIP."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import platform
import shutil
import sys
import tempfile
import zipfile
from contextlib import redirect_stdout, redirect_stderr

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from nbconvert import HTMLExporter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DEST = ROOT / "entrega/Hackaton11_v1"
ZIP = DEST.parent / "Hackaton11_v1_entrega.zip"
members = [DEST / "hack11_v1_ejecutado.ipynb", DEST / "requirements.txt", DEST / "README.md"]
members += sorted((DEST / "data").iterdir()) + sorted((DEST / "resultados").iterdir())
preflight_zip = REPO / "tmp/s13-v1-preflight.zip"
with zipfile.ZipFile(preflight_zip,"w",zipfile.ZIP_DEFLATED) as z:
    for path in members:
        z.write(path,arcname="Hackaton11_v1/"+path.relative_to(DEST).as_posix())
    assert z.testzip() is None
check_root = Path(tempfile.mkdtemp(prefix="s13-v1-zip-",dir=REPO / "tmp")).resolve()
assert check_root.parent == (REPO / "tmp").resolve()
with zipfile.ZipFile(preflight_zip) as z:
    assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in z.namelist())
    z.extractall(check_root)
package = check_root / "Hackaton11_v1"
path = package / "hack11_v1_ejecutado.ipynb"
notebook = nbformat.read(path,4)
nbformat.validate(notebook)
sources_before = [c.source for c in notebook.cells]
code_text = "\n".join(c.source for c in notebook.cells if c.cell_type=="code")
assert "C:\\Users\\" not in code_text and "/kaggle/input" not in code_text
assert "pd.read_csv" not in notebook.cells[36].source
source_original = ROOT / "kaggle/v1/outputs_revision1/hack11_v1_ejecutado.ipynb"
original_digest = hashlib.sha256(source_original.read_bytes()).hexdigest()
log_path = DEST.parent / "verificacion_v1.log"
print("Ejecutando copia extraída del ZIP en entorno aislado...",flush=True)
with log_path.open("w",encoding="utf-8",buffering=1) as log:
    with redirect_stdout(log),redirect_stderr(log):
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
        def start(cell,cell_index,**kwargs):
            print("CELL",cell_index,"START",flush=True)
        try:
            NotebookClient(notebook,km=manager,timeout=600,
                resources={"metadata":{"path":str(package)}},on_cell_start=start).execute()
        finally:
            nbformat.write(notebook,path)
code_cells = [c for c in notebook.cells if c.cell_type=="code"]
assert all(c.execution_count is not None for c in code_cells)
assert not any(o.output_type=="error" for c in code_cells for o in c.outputs)
assert [c.source for c in notebook.cells] == sources_before
summary = json.loads((package / "salidas/run_summary.json").read_text(encoding="utf-8"))
assert summary["status"]=="complete" and summary["stage"]=="done"
actual = pd.read_csv(package / "salidas/tabla_resultados.csv").set_index("Modelo").sort_index()
reference = pd.read_csv(package / "resultados/tabla_resultados.csv").set_index("Modelo").sort_index()
columns = ["F1 - Sólo IA","F1 - IA + Human","Mejora","HACS","Coverage","N_IA","N_Junior","N_Senior","Coste_IA","Coste_colab","F1_colab_MC_mean"]
assert set(actual.index)=={"XGBoost","LightGBM","CatBoost"}
np.testing.assert_allclose(actual[columns],reference[columns],rtol=0,atol=1e-9)
pd.testing.assert_frame_equal(pd.read_csv(package / "salidas/particiones.csv"),pd.read_csv(package / "resultados/particiones.csv"))
truth = np.array([int(line.split()[-1])==2 for line in (package / "data/german.data").read_text().splitlines()],dtype=int)
decisions = pd.read_csv(package / "salidas/decisiones_prueba.csv")
assert len(decisions)==600
for name,d in decisions.groupby("Modelo"):
    assert len(d)==200 and d.row_id.nunique()==200
    assert np.array_equal(d.y_true,truth[d.row_id])
    def f1(pred):
        tp = int(((d.y_true==1)&(pred==1)).sum())
        fp = int(((d.y_true==0)&(pred==1)).sum())
        fn = int(((d.y_true==1)&(pred==0)).sum())
        return 2*tp/(2*tp+fp+fn)
    assert np.isclose(actual.loc[name,"F1 - Sólo IA"],f1(d.y_ai))
    assert np.isclose(actual.loc[name,"F1 - IA + Human"],f1(d.y_final))
    coverage = float((d.responsable=="IA").mean())
    assert np.isclose(actual.loc[name,"HACS"],.6*f1(d.y_final)+.4*coverage)
assert hashlib.sha256(source_original.read_bytes()).hexdigest() == original_digest
for name in ["german.data","readme.md","german.doc"]:
    assert (package / "data" / name).read_bytes() == (ROOT / "data" / name).read_bytes()
report = {"status":"passed","executed_from_extracted_zip":True,"isolated_environment":True,
    "python":platform.python_version(),"platform":platform.platform(),"mode":"local CPU",
    "code_cells_executed":len(code_cells),"runtime_errors":0,"matches_kaggle_v1_metrics":True,
    "metric_tolerance":1e-9,"same_partitions":True,"data_sha256":hashlib.sha256((package/"data/german.data").read_bytes()).hexdigest(),
    "original_notebook_unchanged":True,"versions":summary["versions"],
    "verified_at_utc":datetime.now(timezone.utc).isoformat(),
    "models":actual.reset_index()[["Modelo",*columns]].to_dict("records")}
(DEST / "VALIDACION.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
shutil.copy2(path,DEST / path.name)
html,_ = HTMLExporter().from_notebook_node(notebook)
(DEST.parent / "vista_previa_v1.html").write_text(html,encoding="utf-8")
# Keep new run artifacts for QA, outside the delivery ZIP.
(DEST.parent / "ruta_verificacion.txt").write_text(str(package),encoding="utf-8")
members += [DEST / "VALIDACION.json"]
with zipfile.ZipFile(ZIP,"w",zipfile.ZIP_DEFLATED) as z:
    for file in members:
        z.write(file,arcname="Hackaton11_v1/"+file.relative_to(DEST).as_posix())
    assert z.testzip() is None
    assert len(z.namelist())==len(members)
with zipfile.ZipFile(ZIP) as z:
    for file in members:
        assert z.read("Hackaton11_v1/"+file.relative_to(DEST).as_posix()) == file.read_bytes()
print(json.dumps(report,indent=2,ensure_ascii=False))
print("ZIP:",ZIP,"| bytes:",ZIP.stat().st_size,"| archivos:",len(members),flush=True)
