"""Audit good-credit F1, strict gains, selection and v1 comparisons from saved rows."""
from pathlib import Path
import hashlib
import json
import nbformat
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "outputs"


def f1(y, pred, positive=1):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(((y == positive) & (pred == positive)).sum())
    fp = int(((y != positive) & (pred == positive)).sum())
    fn = int(((y == positive) & (pred != positive)).sum())
    return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0


def cost(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return int(5*((y == 0) & (pred == 1)).sum()+((y == 1) & (pred == 0)).sum())


def main():
    summary = json.loads((OUT/"run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and summary["stage"] == "done"
    assert summary["positive_class"] == "good_credit"
    assert summary["minimum_absolute_f1_gain_strict"] == .01
    raw = (ROOT/"data/german.data").read_bytes()
    assert summary["dataset"]["sha256"] == hashlib.sha256(raw).hexdigest()
    original = np.array([int(line.split()[-1]) for line in raw.decode().splitlines()])
    good = (original == 1).astype(int)
    splits = pd.read_csv(OUT/"particiones.csv")
    v1splits = pd.read_csv(ROOT/"kaggle/v1/outputs/particiones.csv")
    assert np.array_equal(splits.row_id, v1splits.row_id)
    assert np.array_equal(splits.split, v1splits.split)
    assert np.array_equal(splits.y, 1-v1splits.y)
    assert np.array_equal(splits.y, good[splits.row_id])
    assert splits.split.value_counts().to_dict() == {"train":600,"validation":200,"test":200}
    train_ids = set(splits.loc[splits.split=="train","row_id"])
    test_ids = set(splits.loc[splits.split=="test","row_id"])
    folds = pd.read_csv(OUT/"folds_entrenamiento.csv")
    assert len(folds) == 600 and folds.row_id.nunique() == 600
    assert set(folds.row_id) == train_ids and set(folds.fold) == set(range(5))
    oof = pd.read_csv(OUT/"probabilidades_oof_entrenamiento.csv")
    assert len(oof) == 3000 and oof.Modelo.nunique() == 5
    assert set(oof.row_id) == train_ids
    assert not set(oof.row_id) & test_ids
    for name,d in oof.groupby("Modelo"):
        assert len(d) == 600 and d.row_id.nunique() == 600
        assert np.array_equal(d.y_good, good[d.row_id])
    selection = pd.read_csv(OUT/"seleccion_modelos_validacion.csv")
    frozen = json.loads((OUT/"finalistas_congelados.json").read_text())
    eligible = selection[selection.Elegible_validacion].sort_values(["HACS","Coverage"],ascending=False)
    assert frozen["finalists"] == eligible.head(3).Modelo.tolist()
    assert frozen["selected"] == frozen["finalists"][0]
    assert summary["finalists"] == frozen["finalists"]
    threshold_grid = np.unique(np.r_[np.linspace(.1,.9,81),.5])
    for name,d in oof.groupby("Modelo"):
        scores = np.array([f1(d.y_good,d.p_good_oof>=t) for t in threshold_grid])
        tied = threshold_grid[np.isclose(scores,scores.max(),rtol=0,atol=1e-12)]
        chosen_threshold = float(tied[np.argmin(abs(tied-.5))])
        assert np.isclose(frozen["classification_thresholds"][name],chosen_threshold)
    search_policies = pd.read_csv(OUT/"busqueda_umbrales_validacion.csv")
    for name,d in search_policies.groupby("Modelo"):
        feasible = d[d.Elegible_validacion]
        best = (feasible if len(feasible) else d).sort_values(["HACS","Coverage"],ascending=False).iloc[0]
        saved = selection.set_index("Modelo").loc[name]
        assert np.allclose(saved[["A","B","HACS"]].astype(float),best[["A","B","HACS"]].astype(float))
    table = pd.read_csv(OUT/"tabla_resultados.csv")
    decisions = pd.read_csv(OUT/"decisiones_prueba.csv")
    sims = pd.read_csv(OUT/"simulaciones_prueba.csv")
    assert table.Modelo.tolist() == frozen["finalists"] and len(table) == 3
    assert len(decisions) == 600 and len(sims) == 3000
    records = []
    for row in table.to_dict("records"):
        name = row["Modelo"]
        d = decisions[decisions.Modelo==name]
        assert len(d) == 200 and set(d.row_id) == test_ids and d.row_id.nunique() == 200
        assert np.array_equal(d.y_true, good[d.row_id])
        assert np.array_equal(d.y_ai, (d.p_good>=d.threshold).astype(int))
        assert np.allclose(d.threshold, frozen["classification_thresholds"][name])
        p = np.clip(d.p_good.to_numpy(),1e-12,1-1e-12)
        entropy = -p*np.log(p)-(1-p)*np.log(1-p)
        assert np.allclose(d.entropy,entropy)
        pol = selection.set_index("Modelo").loc[name]
        assert pol.Mejora >= .02 and pol.Delta_p10 > .01
        responsible = np.where(entropy<pol.A,"IA",np.where(entropy<pol.B,"Junior","Senior"))
        at_boundary = (abs(entropy-pol.A)<1e-12)|(abs(entropy-pol.B)<1e-12)
        assert np.array_equal(d.responsable.to_numpy()[~at_boundary],responsible[~at_boundary])
        final = np.where(d.responsable=="IA",d.y_ai,np.where(d.responsable=="Junior",d.y_junior,d.y_senior))
        assert np.array_equal(d.y_final,final)
        baseline,collaborative = f1(d.y_true,d.y_ai),f1(d.y_true,d.y_final)
        delta = collaborative-baseline
        coverage = float((d.responsable=="IA").mean())
        efficiency = 1-float((d.responsable!="IA").mean())
        values = {"F1 - Sólo IA":baseline,"F1 - IA + Human":collaborative,"Mejora":delta,
            "Mejora_pp":100*delta,"Mejora_relativa_pct":100*delta/baseline,
            "HACS":.6*collaborative+.2*coverage+.2*efficiency,"Coverage":coverage,"Efficiency":efficiency,
            "F1_bad_IA":f1(d.y_true,d.y_ai,0),"F1_bad_colab":f1(d.y_true,d.y_final,0),
            "Coste_IA":cost(d.y_true,d.y_ai),"Coste_colab":cost(d.y_true,d.y_final)}
        for key,value in values.items():
            assert np.isclose(row[key],value),(name,key,row[key],value)
        assert sum(row[k] for k in ["N_IA","N_Junior","N_Senior"]) == 200
        s = sims[sims.Modelo==name]
        assert len(s)==1000 and s.replica.nunique()==1000
        assert np.allclose(s.HACS,.6*s.F1+.4*coverage)
        assert np.allclose(s.Mejora,s.F1-baseline)
        assert np.isclose(s.iloc[0].F1,collaborative)
        assert np.isclose(row["F1_colab_MC_mean"],s.F1.mean())
        assert np.isclose(row["Mejora_MC_mean"],s.Mejora.mean())
        assert bool(row["Cumple_mayor_1pp"]) == (delta > .01)
        assert bool(row["Cumple_MC_mayor_1pp"]) == (s.Mejora.mean() > .01)
        records.append({"model":name,"f1_ai_good":baseline,"f1_collaborative_good":collaborative,
            "gain_pp":100*delta,"mean_gain_pp":100*s.Mejora.mean(),"hacs":values["HACS"],
            "coverage":coverage,"strict_gain_passed":bool(delta>.01 and s.Mejora.mean()>.01)})
    source = nbf_read(HERE/"input/hack11_v2.ipynb",executed=False)
    remote = nbf_read(OUT/"hack11_v2_ejecutado.ipynb",executed=True)
    assert [c.source for c in source.cells] == [c.source for c in remote.cells]
    download = json.loads((OUT/"notebook_download.json").read_text())
    assert download["kernel"] == "jeffreyamc/hack11-human-ai-credit-v2-good-credit"
    assert download["sha256"] == hashlib.sha256((OUT/"hack11_v2_ejecutado.ipynb").read_bytes()).hexdigest()
    cases = pd.read_csv(OUT/"casos_shap.csv")
    assert len(cases)==9 and cases.row_id.nunique()==9
    assert sorted(cases.groupby("grupo").size()) == [3,3,3]
    assert summary["shap_additivity_max_error"] < 1e-4
    assert len(list(OUT.glob("*.png")))==10
    for entry in json.loads((OUT/"manifest.json").read_text()):
        path = OUT/entry["name"]
        assert path.stat().st_size == entry["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"],path
    historical = []
    for environment,path in [("Kaggle",ROOT/"kaggle/v1/outputs/decisiones_prueba.csv"),
                             ("Local",ROOT/"local/outputs/decisiones_prueba.csv")]:
        for name,d in pd.read_csv(path).groupby("Modelo"):
            assert np.array_equal(1-d.y_true,good[d.row_id])
            baseline = f1(1-d.y_true,1-d.y_ai)
            final = f1(1-d.y_true,1-d.y_final)
            coverage = float((d.responsable=="IA").mean())
            historical.append({"Version":"v1 recodificada","Entorno":environment,"Modelo":name,
                "F1_good_IA":baseline,"F1_good_colab":final,"Mejora":final-baseline,
                "Coverage":coverage,"HACS":.6*final+.4*coverage})
    historical.extend({"Version":"v2","Entorno":"Kaggle","Modelo":r["model"],
        "F1_good_IA":r["f1_ai_good"],"F1_good_colab":r["f1_collaborative_good"],
        "Mejora":r["gain_pp"]/100,"Coverage":r["coverage"],"HACS":r["hacs"]} for r in records)
    historical = pd.DataFrame(historical)
    historical.to_csv(HERE/"comparacion_v1_v2_misma_clase.csv",index=False)
    report = {"status":"passed","positive_class":"good_credit","same_v1_splits":True,
        "cv_training_only":True,"five_candidate_families":True,"three_selected_by_validation_hacs":True,
        "executed_notebook_matches_source":True,"remote_version":download["version"],
        "code_cells":sum(c.cell_type=="code" for c in remote.cells),"cases_shap":9,"figures":10,
        "models":records,"all_three_strict_gain_passed":all(r["strict_gain_passed"] for r in records),
        "best_v1_hacs_same_good_class":float(historical[historical.Version=="v1 recodificada"].HACS.max()),
        "best_v2_hacs":float(table.HACS.max()),"selected_model":frozen["selected"]}
    report["hacs_improved_over_best_v1_same_class"] = report["best_v2_hacs"] > report["best_v1_hacs_same_good_class"]
    (HERE/"validation_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    print(historical.to_string(index=False))
    assert report["all_three_strict_gain_passed"],"Al menos un modelo no mejora >1pp. No declarar objetivo completo."


def nbf_read(path,executed):
    nb = nbformat.read(path,4)
    nbformat.validate(nb)
    counts = []
    for cell in nb.cells:
        if cell.cell_type=="code":
            compile(cell.source,str(path),"exec")
            if executed:
                assert cell.execution_count is not None
                counts.append(cell.execution_count)
            assert not any(o.output_type=="error" for o in cell.get("outputs",[]))
    if executed:
        assert counts == list(range(1,len(counts)+1))
    return nb


if __name__ == "__main__":
    main()
