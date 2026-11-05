"""Independently reconcile the experiment artifacts using counts, not model code."""
import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def binary_f1(truth, pred):
    truth, pred = np.asarray(truth), np.asarray(pred)
    tp = sum((truth == 1) & (pred == 1))
    fp = sum((truth == 0) & (pred == 1))
    fn = sum((truth == 1) & (pred == 0))
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def validate(folder, expected_models):
    summary = json.loads((folder / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and summary["stage"] == "done", summary
    assert summary["dataset"]["sha256"] == hashlib.sha256((ROOT / "data/german.data").read_bytes()).hexdigest()
    decisions = pd.read_csv(folder / "decisiones_prueba.csv")
    table = pd.read_csv(folder / "tabla_resultados.csv")
    policies = pd.read_csv(folder / "politicas_congeladas.csv").set_index("Modelo")
    split = pd.read_csv(folder / "particiones.csv")
    source_y = np.array([int(line.split()[-1]) == 2
                         for line in (ROOT / "data/german.data").read_text().splitlines()], dtype=int)
    assert np.array_equal(split.y, source_y[split.row_id])
    assert set(table.Modelo) == set(expected_models)
    assert split.row_id.nunique() == 1000 and len(split) == 1000
    assert split.split.value_counts().to_dict() == {"train": 600, "validation": 200, "test": 200}
    assert pd.crosstab(split.split, split.y).to_dict() == {
        0: {"test": 140, "train": 420, "validation": 140},
        1: {"test": 60, "train": 180, "validation": 60},
    }
    records = []
    for row in table.to_dict("records"):
        d = decisions[decisions.Modelo == row["Modelo"]]
        assert len(d) == 200 and d.row_id.nunique() == 200
        assert set(d.row_id) == set(split.loc[split.split == "test", "row_id"])
        assert np.array_equal(d.y_true, source_y[d.row_id])
        p = np.clip(d.p_bad.to_numpy(), 1e-12, 1-1e-12)
        h = -p*np.log(p) - (1-p)*np.log(1-p)
        assert np.allclose(d.entropy, h)
        assert np.array_equal(d.y_ai, (d.p_bad >= 0.5).astype(int))
        a, b = policies.loc[row["Modelo"], ["A", "B"]]
        owner = np.where(h < a, "IA", np.where(h < b, "Junior", "Senior"))
        # CSV rounding may move a boundary by machine precision; compare at tolerance.
        boundary = (np.abs(h-a) < 1e-12) | (np.abs(h-b) < 1e-12)
        assert np.all(d.responsable.to_numpy()[~boundary] == owner[~boundary])
        final = np.where(d.responsable == "IA", d.y_ai,
                         np.where(d.responsable == "Junior", d.y_junior, d.y_senior))
        assert np.array_equal(d.y_final, final)
        f_ai, f_final = binary_f1(d.y_true, d.y_ai), binary_f1(d.y_true, d.y_final)
        coverage = sum(d.responsable == "IA")/len(d)
        efficiency = 1-sum(d.responsable != "IA")/len(d)
        hacs = .6*f_final+.2*coverage+.2*efficiency
        for column, expected in {"F1 - Sólo IA": f_ai, "F1 - IA + Human": f_final,
                                 "Mejora": f_final-f_ai, "Mejora_pp": 100*(f_final-f_ai),
                                 "Coverage": coverage, "Efficiency": efficiency, "HACS": hacs}.items():
            assert np.isclose(row[column], expected), (row["Modelo"], column)
        for name, pred in [("Coste_IA", d.y_ai), ("Coste_colab", d.y_final)]:
            cost = int(sum((d.y_true == 0) & (pred == 1)) + 5*sum((d.y_true == 1) & (pred == 0)))
            assert row[name] == cost
        sims = pd.read_csv(folder / "simulaciones_prueba.csv")
        s = sims[sims.Modelo == row["Modelo"]]
        assert len(s) == 500 and s.replica.nunique() == 500
        assert np.allclose(s.HACS, .6*s.F1+.4*coverage)
        assert np.isclose(row["F1_colab_MC_mean"], s.F1.mean())
        assert np.isclose(row["HACS_MC_mean"], s.HACS.mean())
        assert np.isclose(s.iloc[0].F1, f_final)
        records.append({"model": row["Modelo"], "f1_ai": f_ai, "f1_collaborative": f_final, "hacs": hacs})
    cases = pd.read_csv(folder / "casos_shap.csv")
    assert len(cases) == 9 and cases.row_id.nunique() == 9
    assert sorted(cases.groupby("grupo").size()) == [3, 3, 3]
    assert len(list(folder.glob("*.png"))) == 10
    assert summary["shap_additivity_max_error"] < 1e-4
    for entry in json.loads((folder / "manifest.json").read_text(encoding="utf-8")):
        file = folder / entry["name"]
        assert file.stat().st_size == entry["bytes"]
        assert hashlib.sha256(file.read_bytes()).hexdigest() == entry["sha256"], file
    return {"status": "passed", "models": records, "selected_model": summary["selected_model"],
            "rows_per_model": 200, "simulation_repetitions": 500, "shap_cases": 9, "figures": 10}


def validate_notebook(path, require_executed=True):
    notebook = nbformat.read(path, 4)
    nbformat.validate(notebook)
    counts = []
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(path), "exec")
            if require_executed:
                assert cell.execution_count is not None, cell.id
                counts.append(cell.execution_count)
            assert not any(out.output_type == "error" for out in cell.get("outputs", [])), cell.id
    if require_executed:
        assert counts == list(range(1, len(counts)+1)), counts
    return {"path": str(path.relative_to(ROOT)), "cells": len(notebook.cells),
            "executed_code_cells": len(counts), "errors": 0}


if __name__ == "__main__":
    report = {
        "local": validate(ROOT / "local/outputs", ["Random Forest", "Logistic Regression"]),
        "kaggle": validate(ROOT / "kaggle/v1/outputs", ["XGBoost", "LightGBM", "CatBoost"]),
        "notebooks": [validate_notebook(ROOT / "local/hack11_local_v1.ipynb"),
                      validate_notebook(ROOT / "kaggle/v1/outputs/hack11_v1_ejecutado.ipynb"),
                      validate_notebook(ROOT / "kaggle/v1/input/hack11_v1.ipynb", require_executed=False)],
    }
    local_split = pd.read_csv(ROOT / "local/outputs/particiones.csv")
    remote_split = pd.read_csv(ROOT / "kaggle/v1/outputs/particiones.csv")
    pd.testing.assert_frame_equal(local_split, remote_split)
    local_nb = nbformat.read(ROOT / "local/hack11_local_v1.ipynb", 4)
    source_nb = nbformat.read(ROOT / "kaggle/v1/input/hack11_v1.ipynb", 4)
    remote_nb = nbformat.read(ROOT / "kaggle/v1/outputs/hack11_v1_ejecutado.ipynb", 4)
    assert [c.source for c in source_nb.cells] == [c.source for c in remote_nb.cells]
    differing_cells = [i for i, (a, b) in enumerate(zip(local_nb.cells, source_nb.cells)) if a.source != b.source]
    assert differing_cells == [0, 4, 11, 12], differing_cells
    download = json.loads((ROOT / "kaggle/v1/outputs/notebook_download.json").read_text())
    assert download["version"] == 3 and download["errors"] == 0
    assert hashlib.sha256((ROOT / "kaggle/v1/outputs/hack11_v1_ejecutado.ipynb").read_bytes()).hexdigest() == download["sha256"]
    upload = json.loads((ROOT / "kaggle/dataset_upload.json").read_text(encoding="utf-8"))
    assert upload["is_private"] and all(f["exact_match"] for f in upload["files"])
    report["same_splits"] = True
    report["remote_notebook_matches_source"] = True
    report["shared_protocol_sources_match"] = True
    report["remote_version"] = download["version"]
    report["dataset_private_and_verified"] = True
    (ROOT / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
