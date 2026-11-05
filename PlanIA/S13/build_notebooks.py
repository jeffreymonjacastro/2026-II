"""Generate two self-contained, section-aligned Hack11 notebooks."""
from pathlib import Path
import json
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parent


def build(mode):
    cells = []

    def md(source):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))

    def code(source):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))

    models = "XGBoost, LightGBM y CatBoost" if mode == "kaggle" else "Random Forest y Logistic Regression"
    md(f"""
    # Hackatón 11: ¿Confiar, verificar o escalar? — v1 ({mode})

    **Modelos:** {models}. **Equipo:** completar nombres antes de entregar.

    Experimento reproducible de colaboración Humano–IA sobre Statlog German Credit.
    Las seis partes siguen `Sem13_Hackaton11.pdf`. Los humanos son **simulados**,
    no participantes reales. Las conclusiones se generan a partir de esta ejecución.

    **Protocolo:** entrenamiento 60%, validación 20% y prueba 20%, estratificados.
    Elegimos el modelo por F1 de validación y congelamos la política antes de prueba.
    Se comparan todos los candidatos brevemente; el análisis detallado y la defensa
    corresponden exclusivamente al mejor modelo de validación.

    **Fuente:** Hofmann (1994), [UCI Statlog German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data),
    DOI [10.24432/C5NC77](https://doi.org/10.24432/C5NC77), CC BY 4.0.
    Se usa `german.data` (20 atributos); `german.data-numeric` es otra representación,
    **no** un conjunto de observaciones adicionales.
    """)
    md("""
    ### Preparación del entorno

    El notebook es autocontenido. Usa NumPy, pandas, scikit-learn, SciPy, Matplotlib,
    SHAP y joblib. Kaggle incluye además xgboost, lightgbm y catboost. Si falta SHAP,
    instalar `pip install shap` en el entorno antes de ejecutar. No se requieren claves
    dentro del notebook. La sesión remota solicita Tesla T4; se registra el hardware,
    y se usa CPU para estos datos pequeños para mantener el mismo protocolo reproducible.
    """)
    code('''
    import os, sys, json, hashlib, platform, subprocess, time
    from pathlib import Path
    import importlib.metadata as metadata
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown
    from scipy.stats import mannwhitneyu
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
        RocCurveDisplay, brier_score_loss)
    import joblib
    import shap
    plt.rcParams.update({"figure.figsize": (9, 4.5), "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    COLORS = {"IA": "#28689a", "Humano–IA": "#b47722", "Junior": "#7d8050", "Senior": "#a34f80"}
    ''')
    md("""
    ### Configuración, rutas y trazabilidad

    La semilla y los índices de cada partición son compartidos por las dos variantes.
    Se guardan probabilidades, decisiones, búsqueda de umbrales, métricas, gráficos,
    modelo seleccionado y versiones de paquetes. `run_summary.json` empieza como
    `running` y solamente pasa a `complete` al terminar todas las verificaciones.
    """)
    code(f'''
    MODE = {mode!r}
    SEED = 42
    N_VAL_SIM, N_TEST_SIM = 200, 500
    JUNIOR_ACCURACY, SENIOR_ACCURACY = 0.70, 0.95
    GROUP = "Equipo por completar"
    if MODE == "kaggle":
        DATA = next(p.parent for p in Path("/kaggle/input").rglob("german.data"))
        OUT = Path("/kaggle/working")
    else:
        here = Path.cwd().resolve()
        ROOT = next(p for p in [here, *here.parents] if (p / "data/german.data").exists())
        DATA, OUT = ROOT / "data", ROOT / "local/outputs"
    OUT.mkdir(parents=True, exist_ok=True)
    versions = {{p: metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "shap"]}}
    summary = {{"status": "running", "stage": "setup", "mode": MODE, "seed": SEED,
               "python": sys.version, "platform": platform.platform(), "versions": versions,
               "human_simulation_only": True}}
    def checkpoint(stage, **values):
        summary.update(stage=stage, **values)
        (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    def savefig(name):
        plt.tight_layout()
        plt.savefig(OUT / name, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip() if MODE == "kaggle" else "CPU local"
    checkpoint("setup", hardware=gpu, estimator_device="CPU (dataset pequeño)")
    print("Entorno:", MODE, "| Hardware:", gpu, "| Salida:", OUT)
    ''')
    md("""
    ## Parte 1 — Comprensión del problema y exploración de datos (10%)

    ### Actividad 1.1 — Comprensión del problema

    Aprobar a una persona que incumple puede causar pérdida de capital y gastos de
    recuperación. Rechazar a un buen cliente implica ingresos perdidos, daño a la relación
    comercial y posibles perjuicios injustos. Una clasificación automática también puede
    reproducir sesgos presentes en los datos.

    El README fija costes diferentes: **mal cliente aprobado = 5** y **buen cliente
    rechazado = 1**. Son costes relativos, no dinero real. Codificamos `y=1` como **mal
    crédito** (código original 2), por lo que un falso negativo cuesta cinco y un falso
    positivo cuesta uno. El F1 principal es binario para mal crédito; también reportamos
    F1 macro. El PDF no especifica la variante de F1, por eso esta elección es explícita.

    ### Actividad 1.2 — Carga y comprobación del dataset

    Usamos las 20 columnas originales y su diccionario local. Las categorías A65 y A124
    describen estados desconocidos en el origen; no son celdas nulas ni se borran.
    Las columnas personales sensibles se conservan para reproducir el ejercicio, sin
    interpretar el desempeño como evidencia suficiente de equidad o aptitud bancaria.
    """)
    code('''
    FEATURES = ["checking", "duration", "credit_history", "purpose", "amount", "savings",
                "employment", "installment_rate", "personal_status_sex", "guarantors",
                "residence_since", "property", "age", "other_plans", "housing",
                "existing_credits", "job", "dependents", "telephone", "foreign_worker"]
    NUM = ["duration", "amount", "installment_rate", "residence_since", "age", "existing_credits", "dependents"]
    CAT = [c for c in FEATURES if c not in NUM]
    df = pd.read_csv(DATA / "german.data", sep=r"\\s+", header=None, names=FEATURES + ["original_class"])
    assert df.shape == (1000, 21)
    assert set(df.original_class) == {1, 2}
    X = df[FEATURES].copy()
    y = (df.original_class == 2).astype(int).to_numpy()
    assert np.bincount(y).tolist() == [700, 300]
    digest = hashlib.sha256((DATA / "german.data").read_bytes()).hexdigest()
    quality = {"rows": len(df), "features": len(FEATURES), "numeric": len(NUM), "categorical": len(CAT),
               "missing_cells": int(df.isna().sum().sum()), "duplicate_rows": int(df.duplicated().sum()),
               "sha256": digest, "class_0_good": int(sum(y == 0)), "class_1_bad": int(sum(y == 1))}
    checkpoint("data", dataset=quality)
    display(pd.Series(quality, name="Valor").to_frame())
    display(df.head(4))
    display(df[NUM].describe().round(2))
    display(pd.DataFrame({"variable": CAT, "categorias": [df[c].nunique() for c in CAT]}))
    ''')
    md("""
    ### Exploración gráfica y correlaciones

    Edad, duración y balance de clases describen la población. Las correlaciones son
    Pearson para variables numéricas y la etiqueta binaria; no implican causalidad.
    Este EDA no interviene en la selección de atributos, umbrales ni modelos. El conjunto
    de prueba se mantiene fuera de toda decisión de ajuste.
    """)
    code('''
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].hist(df.age, bins=18, color=COLORS["IA"], edgecolor="white")
    axes[0].set(xlabel="Edad (años)", ylabel="Solicitantes", title="Edad")
    axes[1].hist(df.duration, bins=16, color=COLORS["IA"], edgecolor="white")
    axes[1].set(xlabel="Duración (meses)", ylabel="Solicitantes", title="Duración del crédito")
    axes[2].bar(["Bueno", "Malo"], np.bincount(y), color=[COLORS["IA"], COLORS["Humano–IA"]])
    axes[2].set(ylabel="Solicitantes", title="Clases originales (n = 1,000)")
    savefig("01_exploracion.png")
    corr = df[NUM].assign(bad_credit=y).corr()
    corr.to_csv(OUT / "correlaciones.csv")
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="BrBG", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Correlación Pearson")
    ax.set_title("Variables numéricas y mal crédito · n = 1,000")
    savefig("02_correlaciones.png")
    display(corr["bad_credit"].drop("bad_credit").sort_values(key=abs, ascending=False).to_frame().round(3))
    ''')
    md("""
    ## Parte 2 — Desarrollo del modelo base (20%)

    ### Actividad 2.1 — Preprocesamiento y particiones

    Estratificamos 600/200/200 solicitudes conservando 70%/30% en cada partición.
    Ajustamos imputación, one-hot y escalamiento **sólo en entrenamiento**. La mediana
    y la moda son medidas preventivas ante nulos; no se crean nulos artificiales.
    Se escalan los siete atributos numéricos, necesario para regresión logística y
    inocuo para árboles. One-hot evita imponer un orden ficticio a códigos categóricos.
    No usamos SMOTE ni alteramos la prevalencia de validación o prueba.
    """)
    code('''
    idx_trainval, idx_test = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=SEED)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.25, stratify=y[idx_trainval], random_state=SEED)
    assert not (set(idx_train) & set(idx_val) or set(idx_train) & set(idx_test) or set(idx_val) & set(idx_test))
    split = pd.DataFrame({"row_id": np.arange(len(df)), "split": "train", "y": y})
    split.loc[idx_val, "split"] = "validation"
    split.loc[idx_test, "split"] = "test"
    split.to_csv(OUT / "particiones.csv", index=False)
    display(pd.crosstab(split.split, split.y).rename(columns={0: "Bueno", 1: "Malo"}))
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT)])
    Ztrain = pre.fit_transform(X.iloc[idx_train])
    Zval, Ztest = pre.transform(X.iloc[idx_val]), pre.transform(X.iloc[idx_test])
    yt, yv, ye = y[idx_train], y[idx_val], y[idx_test]
    names = pre.get_feature_names_out()
    print("Matrices transformadas:", Ztrain.shape, Zval.shape, Ztest.shape)
    ''')
    md(f"""
    ### Actividad 2.2 — Entrenamiento de {models}

    Comparación controlada con hiperparámetros modestos, semilla fija y umbral de
    clasificación 0.5. No son probabilidades calibradas: guardamos Brier score y
    verificamos empíricamente si la entropía distingue errores. El mejor modelo se
    elige por **F1 binario en validación**, con ROC-AUC como desempate; prueba no
    decide el modelo. Se conserva el ajuste de entrenamiento para que la política
    elegida en validación no cambie al reentrenar.
    """)
    if mode == "kaggle":
        code('''
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
        models = {
            "XGBoost": XGBClassifier(n_estimators=220, max_depth=3, learning_rate=0.04,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=3, eval_metric="logloss", random_state=SEED, n_jobs=2),
            "LightGBM": LGBMClassifier(n_estimators=220, max_depth=3, num_leaves=7,
                learning_rate=0.04, min_child_samples=20, reg_lambda=3, verbosity=-1, random_state=SEED, n_jobs=2),
            "CatBoost": CatBoostClassifier(iterations=300, depth=4, learning_rate=0.04,
                l2_leaf_reg=5, loss_function="Logloss", random_seed=SEED, verbose=False,
                allow_writing_files=False, thread_count=2)}
        for package in ["xgboost", "lightgbm", "catboost"]:
            versions[package] = metadata.version(package)
        ''')
    else:
        code('''
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8,
                min_samples_leaf=3, class_weight="balanced", random_state=SEED, n_jobs=2),
            "Logistic Regression": LogisticRegression(C=1.0, max_iter=2000,
                class_weight="balanced", solver="lbfgs", random_state=SEED)}
        ''')
    md("""
    ### Actividad 2.3 — Métricas y selección sin consultar prueba

    Precision, Recall y F1 consideran positivo al mal crédito. ROC-AUC usa
    probabilidades; el coste relativo complementa las métricas de clasificación.
    La tabla siguiente es de **validación**. Las métricas finales de prueba se
    calculan posteriormente, con decisiones y umbrales congelados.
    """)
    code('''
    def metrics(truth, pred, probability=None):
        tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0,1]).ravel()
        return {"Accuracy": accuracy_score(truth,pred), "Precision": precision_score(truth,pred,zero_division=0),
                "Recall": recall_score(truth,pred,zero_division=0), "F1": f1_score(truth,pred,zero_division=0),
                "F1_macro": f1_score(truth,pred,average="macro",zero_division=0),
                "ROC_AUC": roc_auc_score(truth,probability) if probability is not None else np.nan,
                "Coste": int(fp + 5*fn), "Coste_por_caso": float((fp+5*fn)/len(truth)),
                "TN":int(tn), "FP":int(fp), "FN":int(fn), "TP":int(tp)}
    val_prob, test_prob, training_rows = {}, {}, []
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(Ztrain, yt)
        assert list(model.classes_) == [0,1]
        val_prob[name] = model.predict_proba(Zval)[:,1]
        training_rows.append({"Modelo":name, **metrics(yv,val_prob[name]>=0.5,val_prob[name]),
                              "Brier":brier_score_loss(yv,val_prob[name]), "Segundos":time.perf_counter()-start})
    validation = pd.DataFrame(training_rows).sort_values(["F1","ROC_AUC"],ascending=False)
    best_name = validation.iloc[0].Modelo
    best_model = models[best_name]
    validation.to_csv(OUT / "modelos_validacion.csv",index=False)
    checkpoint("models", selected_model=best_name, versions=versions)
    display(validation.round(4))
    print("Modelo seleccionado sin mirar prueba:", best_name)
    ''')
    md("""
    ## Parte 3 — Cuantificación de incertidumbre (20%)

    ### Actividades 3.1 y 3.2 — Probabilidades y entropía

    Para `p = P(mal crédito | x)`, calculamos
    $H(x)=-p\\ln(p)-(1-p)\\ln(1-p)$, en nats. Vale cero en certeza y alcanza
    $\\ln(2)$ cerca de 0.5. Se recortan probabilidades sólo para evitar `log(0)`.
    **No** confundimos baja entropía con corrección garantizada. En esta sección
    analizamos validación para justificar los umbrales; repetimos el diagnóstico
    en prueba una vez congelada la política.
    """)
    code('''
    def entropy(probability):
        p = np.clip(np.asarray(probability,dtype=float),1e-12,1-1e-12)
        return -p*np.log(p) - (1-p)*np.log(1-p)
    assert np.isclose(entropy([0.5])[0],np.log(2))
    assert entropy([0,1]).max() < 1e-8
    assert np.allclose(entropy([0.2,0.8]), entropy([0.8,0.2]))
    pv = val_prob[best_name]
    uv = entropy(pv)
    uncertainty_val = pd.DataFrame({"row_id":idx_val,"y":yv,"p_bad":pv,"entropy":uv,"correct":(pv>=0.5)==yv})
    uncertainty_val.to_csv(OUT / "incertidumbre_validacion.csv",index=False)
    display(uncertainty_val.head().round(4))
    ''')
    md("""
    ### Actividades 3.3 y 3.4 — ¿La incertidumbre identifica errores?

    Mostramos histograma, boxplot aciertos/errores y accuracy por intervalos de
    entropía. Los intervalos vacíos no se dibujan como accuracy cero. Comparamos
    medias y usamos Mann–Whitney bilateral como análisis exploratorio; AUC de
    detección de error > 0.5 indica que mayor entropía tiende a identificar errores.
    La prueba estadística no demuestra calibración ni determina sola el escalamiento.
    """)
    code('''
    def uncertainty_analysis(prob, truth, split_name, filename):
        u = entropy(prob)
        correct = (prob>=0.5)==truth
        fig, axes = plt.subplots(1,3,figsize=(14,4.2))
        axes[0].hist(u,bins=18,color=COLORS["IA"],edgecolor="white")
        axes[0].set(xlabel="Entropía (nats)",ylabel="Casos",title="Distribución de incertidumbre")
        axes[1].boxplot([u[correct],u[~correct]],tick_labels=["Aciertos","Errores"])
        axes[1].set(ylabel="Entropía (nats)",title="Aciertos frente a errores")
        bins = pd.cut(u,np.linspace(0,np.log(2)+1e-10,9),include_lowest=True)
        grouped = pd.DataFrame({"bin":bins,"correct":correct,"u":u}).groupby("bin",observed=True).agg(
            entropy=("u","mean"),accuracy=("correct","mean"),n=("correct","size"))
        axes[2].plot(grouped.entropy,grouped.accuracy,"o-",color=COLORS["IA"])
        for row in grouped.itertuples():
            axes[2].annotate(str(row.n),(row.entropy,row.accuracy),xytext=(0,5),textcoords="offset points",fontsize=8)
        axes[2].set(xlabel="Entropía media (nats)",ylabel="Accuracy",ylim=(0,1.08),title="Accuracy por intervalo · etiquetas = n")
        fig.suptitle(f"{best_name} · {split_name} (n = {len(truth)})",y=1.03)
        savefig(filename)
        stats = {"split":split_name,"mean_correct":float(u[correct].mean()),"mean_error":float(u[~correct].mean()),
                 "mann_whitney_p":float(mannwhitneyu(u[correct],u[~correct],alternative="two-sided").pvalue),
                 "error_detection_auc":float(roc_auc_score(~correct,u))}
        display(pd.Series(stats).to_frame("Valor"))
        return stats
    uncertainty_stats = [uncertainty_analysis(pv,yv,"validación","03_incertidumbre_validacion.png")]
    ''')
    md("""
    ## Parte 4 — Diseño del sistema Humano–IA (25%)

    ### Actividad 4.1 — Política y justificación de umbrales

    Si `H < A`, decide IA; si `A <= H < B`, revisa junior; si `H >= B`, revisa senior.
    Permitir `A = B` evita forzar trabajo a un junior cuando no aporta calidad.
    Los candidatos son cuantiles de entropía **de validación**, más los extremos.
    La selección detallada se realiza en Parte 6.1 usando únicamente validación.

    ### Actividad 4.2 — Analistas con diferentes grados de instrucción

    Junior: 70% de acierto; senior: 95%, tal como propone la guía. Aquí «precisión»
    del simulador significa **probabilidad de acierto (accuracy)**, no la métrica
    Precision. Cada decisión se genera con una variable Bernoulli: si falla,
    invierte la clase. La simulación consume `y_true` exclusivamente para construir
    respuestas sintéticas; el enrutador jamás recibe etiquetas reales.

    Reutilizamos las mismas respuestas entre políticas para compararlas de forma
    pareada. No reiniciamos el generador dentro del bucle por solicitante, lo que
    crearía errores artificialmente correlacionados. La accuracy humana supuesta
    es constante por clase y dificultad; esta simplificación puede sobreestimar
    ayuda en casos difíciles y no mide automatización sesgada ni fatiga.
    """)
    code('''
    def simulate_analysts(truth, repetitions, seed):
        rng = np.random.default_rng(seed)
        u = rng.random((2,repetitions,len(truth)))
        return (np.where(u[0] < JUNIOR_ACCURACY, truth, 1-truth),
                np.where(u[1] < SENIOR_ACCURACY, truth, 1-truth))
    def route(uncertainty,a,b):
        if not 0 <= a <= b:
            raise ValueError("Los umbrales requieren 0 <= A <= B")
        return np.where(uncertainty<a,"IA",np.where(uncertainty<b,"Junior","Senior"))
    def collaborate(ai,makers,junior,senior):
        return np.where(makers=="IA",ai,np.where(makers=="Junior",junior,senior))
    def f1_many(truth,predictions):
        tp = np.sum((predictions==1)&(truth==1),axis=-1)
        fp = np.sum((predictions==1)&(truth==0),axis=-1)
        fn = np.sum((predictions==0)&(truth==1),axis=-1)
        return np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp,dtype=float),where=(2*tp+fp+fn)>0)
    # Controles con resultados conocidos y extremos de la política.
    assert np.isclose(f1_many(np.array([1,1,0,0]),np.array([[1,0,1,0]]))[0],0.5)
    assert np.all(route(np.array([0.1,0.6]),np.log(2)+1e-8,np.log(2)+1e-8)=="IA")
    assert np.all(route(np.array([0.1,0.6]),0,0)=="Senior")
    jcheck,scheck = simulate_analysts(np.tile([0,1],5000),10,10)
    assert abs(np.mean(jcheck==np.tile([0,1],5000))-0.70)<0.01
    assert abs(np.mean(scheck==np.tile([0,1],5000))-0.95)<0.01
    jval,sval = simulate_analysts(yv,N_VAL_SIM,seed=1001)
    ''')
    md("""
    ### Actividades 4.3 y 4.4 — Pipeline colaborativo y comparación

    El pipeline encadena probabilidades, entropía, responsable y decisión final.
    Una solicitud se cuenta una sola vez como IA, junior o senior. La comparación
    final incluye IA sola, junior para todos, senior para todos y sistema colaborativo.
    Las respuestas simuladas y el responsable quedan guardados por fila para auditar
    F1, cobertura y coste. Primero congelamos los umbrales con la optimización siguiente.

    ## Parte 6.1 — Optimización de la política en validación (10%)

    Presentamos esta actividad antes de SHAP y de evaluar prueba para hacer explícito
    el orden del experimento. Buscamos el mayor HACS medio entre políticas que mejoran
    F1 al menos 0.01 y mejoran en al menos 80% de 200 simulaciones de validación.
    Esos guardarraíles se fijaron antes de ejecutar; no garantizan mejora en prueba.
    Si ninguna política cumple, se conserva la mejor candidata y se marca **no elegible**.

    $Coverage=N_{IA}/N$; $Efficiency=1-(N_{Junior}+N_{Senior})/N$.

    $HACS=0.6\\,F1_{colab}+0.2\\,Coverage+0.2\\,Efficiency$.

    Coverage y Efficiency son iguales en este diseño. Se mantienen ambas columnas
    para seguir el PDF. El score no distingue esfuerzo senior/junior ni incorpora
    la matriz de costes; esos datos se reportan aparte. La mejora es
    $\\Delta F1=F1_{colab}-F1_{IA}$, y `Mejora_pp = 100 × ΔF1`.
    """)
    code('''
    policies, grid_rows = {}, []
    for name,prob in val_prob.items():
        u,ai = entropy(prob),prob>=0.5
        baseline = f1_score(yv,ai)
        thresholds = np.unique(np.r_[0,np.quantile(u,np.linspace(0.1,0.95,18)),np.log(2)+1e-8])
        rows = []
        for a in thresholds:
            for b in thresholds[thresholds>=a]:
                makers = route(u,a,b)
                fs = f1_many(yv,collaborate(ai,makers,jval,sval))
                coverage = np.mean(makers=="IA")
                efficiency = 1-np.mean(makers!="IA")
                mean_f1 = float(fs.mean())
                row = {"Modelo":name,"A":float(a),"B":float(b),"F1_IA":baseline,
                       "F1_colab_mean":mean_f1,"Mejora":mean_f1-baseline,
                       "Prob_mejora":float(np.mean(fs>baseline)),"Coverage":coverage,"Efficiency":efficiency,
                       "HACS":0.6*mean_f1+0.2*coverage+0.2*efficiency,
                       "Junior":int(sum(makers=="Junior")),"Senior":int(sum(makers=="Senior"))}
                row["Elegible_validacion"] = row["Mejora"]>=0.01 and row["Prob_mejora"]>=0.8
                rows.append(row)
        grid = pd.DataFrame(rows)
        feasible = grid[grid.Elegible_validacion]
        choice = (feasible if len(feasible) else grid).sort_values(["HACS","Coverage"],ascending=False).iloc[0].to_dict()
        policies[name] = choice
        grid_rows.extend(rows)
    search = pd.DataFrame(grid_rows)
    search.to_csv(OUT / "busqueda_umbrales_validacion.csv",index=False)
    policy_table = pd.DataFrame(policies.values())
    policy_table.to_csv(OUT / "politicas_congeladas.csv",index=False)
    (OUT / "politicas_congeladas.json").write_text(json.dumps(policies,indent=2),encoding="utf-8")
    display(policy_table.round(4))
    chosen = policies[best_name]
    display(Markdown(f"**Política del modelo elegido:** A={chosen['A']:.4f}, B={chosen['B']:.4f} nats. "
        f"En validación: ΔF1 medio={chosen['Mejora']:.4f}, HACS={chosen['HACS']:.4f}, "
        f"cobertura={chosen['Coverage']:.1%}, junior={chosen['Junior']}, senior={chosen['Senior']}. "
        f"Cumple guardarraíles: **{chosen['Elegible_validacion']}**."))
    selected_grid = search[search.Modelo==best_name]
    fig,ax = plt.subplots(figsize=(9,4.5))
    ax.scatter(selected_grid.Coverage,selected_grid.HACS,s=15,alpha=0.5,color=COLORS["IA"],label="Candidatas")
    ax.scatter([chosen["Coverage"]],[chosen["HACS"]],s=95,marker="*",color=COLORS["Humano–IA"],label="Elegida")
    ax.set(xlabel="Cobertura IA",ylabel="HACS medio",title=f"Políticas en validación · {best_name}",xlim=(0,1),ylim=(0,1))
    ax.legend()
    savefig("04_politicas_validacion.png")
    checkpoint("policy_frozen", policy=chosen, validation_repetitions=N_VAL_SIM)
    ''')
    md("""
    ### Evaluación final congelada — actividades 2.3, 3.4 y 4.4

    Abrimos prueba una vez terminadas todas las elecciones. Reportamos una realización
    reproducible (réplica 0, sin buscar una semilla favorable) y 500 réplicas nuevas.
    El intervalo Monte Carlo 2.5–97.5% mide variabilidad **del simulador humano para
    estos 200 casos**, no incertidumbre de población. Guardamos ambas tablas y las
    decisiones completas. El HACS oficial de una realización y su media no se mezclan.
    """)
    code('''
    jtest,stest = simulate_analysts(ye,N_TEST_SIM,seed=20261104)
    final_rows, sim_rows, decision_frames, base_rows = [], [], [], []
    for name,model in models.items():
        prob = model.predict_proba(Ztest)[:,1]
        test_prob[name] = prob
        ai,u = (prob>=0.5).astype(int),entropy(prob)
        policy = policies[name]
        makers = route(u,policy["A"],policy["B"])
        decisions = collaborate(ai,makers,jtest,stest).astype(int)
        coverage,efficiency = float(np.mean(makers=="IA")),float(1-np.mean(makers!="IA"))
        assert np.isclose(coverage,efficiency)
        fm = f1_many(ye,decisions)
        base = metrics(ye,ai,prob)
        base_rows.append({"Modelo":name,**base,"Brier":brier_score_loss(ye,prob)})
        point = metrics(ye,decisions[0])
        hacs = 0.6*point["F1"]+0.2*coverage+0.2*efficiency
        row = {"Grupo":GROUP,"Modelo":name,"F1 - Sólo IA":base["F1"],"F1 - IA + Human":point["F1"],
               "Mejora":point["F1"]-base["F1"],"Mejora_pp":100*(point["F1"]-base["F1"]),
               "HACS":hacs,"Coverage":coverage,"Efficiency":efficiency,
               "N_IA":int(sum(makers=="IA")),"N_Junior":int(sum(makers=="Junior")),"N_Senior":int(sum(makers=="Senior")),
               "F1_colab_MC_mean":float(fm.mean()),"F1_colab_MC_p025":float(np.quantile(fm,0.025)),
               "F1_colab_MC_p975":float(np.quantile(fm,0.975)),"Prob_mejora_MC":float(np.mean(fm>base["F1"])),
               "HACS_MC_mean":float(0.6*fm.mean()+0.4*coverage),
               "Coste_IA":base["Coste"],"Coste_colab":point["Coste"],
               "Elegible_prueba":bool(point["F1"]>base["F1"]),"Seleccionado_validacion":name==best_name}
        final_rows.append(row)
        sim_rows.extend({"Modelo":name,"replica":r,"F1":float(f),"Mejora":float(f-base["F1"]),
                         "HACS":float(0.6*f+0.4*coverage)} for r,f in enumerate(fm))
        decision_frames.append(pd.DataFrame({"Modelo":name,"row_id":idx_test,"y_true":ye,"p_bad":prob,
            "entropy":u,"y_ai":ai,"responsable":makers,"y_junior":jtest[0],"y_senior":stest[0],"y_final":decisions[0]}))
    results = pd.DataFrame(final_rows)
    all_decisions = pd.concat(decision_frames,ignore_index=True)
    all_decisions.to_csv(OUT / "decisiones_prueba.csv",index=False)
    results.to_csv(OUT / "tabla_resultados.csv",index=False)
    pd.DataFrame(sim_rows).to_csv(OUT / "simulaciones_prueba.csv",index=False)
    pd.DataFrame(base_rows).to_csv(OUT / "modelos_prueba.csv",index=False)
    display(results[["Modelo","F1 - Sólo IA","F1 - IA + Human","Mejora","HACS","Coverage","Elegible_prueba"]].round(4))
    display(results[["Modelo","F1_colab_MC_mean","F1_colab_MC_p025","F1_colab_MC_p975","Prob_mejora_MC"]].round(4))
    selected_decisions = all_decisions[all_decisions.Modelo==best_name].reset_index(drop=True)
    selected_result = results[results.Modelo==best_name].iloc[0]
    comparison = pd.DataFrame([
        {"Sistema":"IA",**metrics(ye,selected_decisions.y_ai,test_prob[best_name])},
        {"Sistema":"Junior",**metrics(ye,jtest[0])},
        {"Sistema":"Senior",**metrics(ye,stest[0])},
        {"Sistema":"Humano–IA",**metrics(ye,selected_decisions.y_final)}])
    comparison.to_csv(OUT / "comparacion_sistemas.csv",index=False)
    display(comparison[["Sistema","Accuracy","Precision","Recall","F1","F1_macro","Coste"]].round(4))
    ''')
    md("""
    ### Matriz de confusión, ROC y comparación de sistemas

    La ROC corresponde a probabilidades del modelo IA. El sistema colaborativo
    entrega decisiones discretas, por lo que no inventamos una curva ROC a partir
    de respuestas humanas. La matriz colaborativa permite revisar cambios en FP/FN
    y reconciliar el coste. Las barras comparan la réplica reproducible, no las medias.
    """)
    code('''
    fig,axes = plt.subplots(1,3,figsize=(14,4))
    ConfusionMatrixDisplay.from_predictions(ye,selected_decisions.y_ai,display_labels=["Bueno","Malo"],ax=axes[0],colorbar=False,cmap="Blues")
    axes[0].set_title("IA sola · prueba")
    ConfusionMatrixDisplay.from_predictions(ye,selected_decisions.y_final,display_labels=["Bueno","Malo"],ax=axes[1],colorbar=False,cmap="Blues")
    axes[1].set_title("Humano–IA · prueba")
    RocCurveDisplay.from_predictions(ye,test_prob[best_name],ax=axes[2],name=best_name)
    axes[2].plot([0,1],[0,1],"--",color="gray")
    axes[2].set_title("ROC IA · positivo = mal crédito")
    savefig("05_evaluacion_prueba.png")
    fig,ax = plt.subplots(figsize=(9,4.5))
    bars = ax.bar(comparison.Sistema,comparison.F1,color=[COLORS[x] for x in comparison.Sistema])
    ax.bar_label(bars,fmt="%.3f",padding=3)
    ax.set(ylabel="F1 binario · mal crédito",ylim=(0,1),title=f"{best_name} · prueba (n = 200), réplica 0")
    savefig("06_comparacion_sistemas.png")
    uncertainty_stats.append(uncertainty_analysis(test_prob[best_name],ye,"prueba","07_incertidumbre_prueba.png"))
    pd.DataFrame(uncertainty_stats).to_csv(OUT / "estadistica_incertidumbre.csv",index=False)
    ''')
    md("""
    ### Sensibilidad a la simulación y a la muestra de prueba

    El histograma describe 500 respuestas humanas alternativas con igual política.
    Además, un bootstrap pareado de solicitudes mantiene juntas las decisiones IA y
    colaborativas de cada caso de la réplica 0. Su intervalo percentil es exploratorio:
    no incluye el entrenamiento, la selección de umbrales ni la simulación humana
    adicional. No es una prueba de mejora garantizada en futuros solicitantes.
    """)
    code('''
    rng = np.random.default_rng(903)
    bootstrap_delta = []
    ya = selected_decisions.y_ai.to_numpy()
    yf = selected_decisions.y_final.to_numpy()
    for _ in range(1000):
        ix = rng.integers(0,len(ye),len(ye))
        bootstrap_delta.append(f1_score(ye[ix],yf[ix],zero_division=0)-f1_score(ye[ix],ya[ix],zero_division=0))
    ci = np.quantile(bootstrap_delta,[0.025,0.975])
    pd.DataFrame({"delta_F1":bootstrap_delta}).to_csv(OUT / "bootstrap_mejora.csv",index=False)
    sim = pd.DataFrame(sim_rows)
    s = sim[sim.Modelo==best_name]
    fig,ax = plt.subplots(figsize=(9,4.5))
    ax.hist(s.Mejora,bins=25,color=COLORS["Humano–IA"],edgecolor="white")
    ax.axvline(0,color="black",linestyle="--",label="Sin mejora")
    ax.set(xlabel="ΔF1 colaborativo − IA",ylabel="Réplicas",title=f"Variabilidad humana simulada · {best_name}, 500 réplicas")
    ax.legend()
    savefig("08_variabilidad_humana.png")
    print("IC bootstrap pareado 95% de ΔF1 (réplica 0):",np.round(ci,4))
    checkpoint("test", test_repetitions=N_TEST_SIM, bootstrap_delta_ci=ci.tolist())
    ''')
    md("""
    ## Parte 5 — Explicabilidad (15%)

    ### Actividad 5.1 — SHAP del modelo seleccionado

    Se explican las variables transformadas del mismo modelo evaluado. Para árboles
    usamos TreeExplainer; para regresión logística, LinearExplainer con fondo sólo de
    entrenamiento. Se normaliza la salida binaria para explicar **mal crédito**.
    Comprobamos aditividad: valor base + suma SHAP reconstruye probabilidad (Random
    Forest) o log-odds (boosting/regresión logística).

    Las importancias globales se agrupan de vuelta a las 20 variables originales.
    El resumen global usa prueba para **interpretación posterior**, sin ajustar modelo
    ni política. SHAP es una atribución del modelo, no evidencia causal ni certificación
    de justicia. Las dependencias entre atributos pueden repartir las atribuciones.

    Documentación: [TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
    y [LinearExplainer](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html).
    """)
    code('''
    if best_name == "Logistic Regression":
        explainer = shap.LinearExplainer(best_model,Ztrain)
        explanation = explainer(Ztest)
        units = "log-odds de mal crédito"
    else:
        explainer = shap.TreeExplainer(best_model)
        explanation = explainer(Ztest)
        units = "probabilidad de mal crédito" if best_name=="Random Forest" else "log-odds de mal crédito"
    values,base = np.asarray(explanation.values),np.asarray(explanation.base_values)
    if values.ndim==3:
        values = values[:,:,1]
        base = base[:,1]
    if base.ndim==0:
        base = np.full(len(ye),float(base))
    reconstructed = base + values.sum(axis=1)
    p = test_prob[best_name]
    expected = p if best_name=="Random Forest" else np.log(np.clip(p,1e-10,1-1e-10)/(1-np.clip(p,1e-10,1-1e-10)))
    additivity_error = float(np.max(np.abs(reconstructed-expected)))
    assert np.allclose(reconstructed,expected,atol=1e-4), additivity_error
    onehot = pre.named_transformers_["cat"].named_steps["onehot"]
    source_features = NUM + [c for c,levels in zip(CAT,onehot.categories_) for _ in levels]
    assert len(source_features)==values.shape[1]
    aggregated = pd.DataFrame(values,columns=source_features).T.groupby(level=0,sort=False).sum().T
    importance = aggregated.abs().mean().sort_values(ascending=False)
    importance.rename("mean_abs_SHAP").to_csv(OUT / "shap_importancia_global.csv")
    pd.DataFrame(values,columns=names).assign(row_id=idx_test).to_csv(OUT / "shap_valores.csv",index=False)
    fig,ax = plt.subplots(figsize=(9,5))
    importance.head(12).sort_values().plot.barh(ax=ax,color=COLORS["IA"])
    ax.set(xlabel=f"Media |SHAP| ({units})",ylabel="Variable original",title=f"Importancia global SHAP · {best_name} · prueba")
    savefig("09_shap_global.png")
    print("Error máximo de aditividad SHAP:",additivity_error,"| Unidades:",units)
    ''')
    md("""
    ### Actividad 5.2 — Nueve casos de estudio

    Seleccionamos tres aciertos entre las menores entropías, tres aciertos distintos
    entre las mayores y tres errores. «Baja» y «alta» son relativas a los aciertos
    observados, no garantizan pertenecer a zonas de enrutamiento; mostramos ambos.
    La selección de casos es ilustrativa, posterior a la evaluación, y no modifica
    decisiones. Si faltaran casos en un grupo, se registra la disponibilidad real.
    """)
    code('''
    correct = selected_decisions.y_ai.to_numpy()==ye
    u = selected_decisions.entropy.to_numpy()
    correct_ids = np.flatnonzero(correct)
    low = correct_ids[np.argsort(u[correct_ids])[:3]]
    remaining = np.array([i for i in correct_ids if i not in low],dtype=int)
    high = remaining[np.argsort(u[remaining])[-3:][::-1]]
    errors = np.flatnonzero(~correct)
    wrong = errors[np.argsort(u[errors])[-3:][::-1]]
    case_groups = {"Correcta / baja entropía":low,"Correcta / alta entropía":high,"Incorrecta":wrong}
    case_rows = []
    for category,ids in case_groups.items():
        for i in ids:
            top = aggregated.iloc[i].abs().nlargest(3).index
            impacts = "; ".join(f"{f}={X.iloc[idx_test[i]][f]}: SHAP {aggregated.iloc[i][f]:+.3f}" for f in top)
            case_rows.append({"grupo":category,"pos_test":int(i),"row_id":int(idx_test[i]),"y_true":int(ye[i]),
                "y_ai":int(ya[i]),"p_bad":float(p[i]),"entropy":float(u[i]),
                "responsable":selected_decisions.iloc[i].responsable,"impactos":impacts})
    cases = pd.DataFrame(case_rows)
    cases.to_csv(OUT / "casos_shap.csv",index=False)
    display(cases[["grupo","row_id","y_true","y_ai","p_bad","entropy","responsable"]].round(4))
    fig,axes = plt.subplots(3,3,figsize=(17,12))
    for ax,case in zip(axes.flat,cases.itertuples()):
        contributions = aggregated.iloc[case.pos_test]
        top = contributions.abs().nlargest(5).index
        v = contributions[top].sort_values()
        ax.barh(v.index,v.values,color=[COLORS["IA"] if z<0 else COLORS["Humano–IA"] for z in v.values])
        ax.axvline(0,color="gray",linewidth=0.8)
        ax.set_title(f"Fila {case.row_id} · real={case.y_true}, IA={case.y_ai}\\nH={case.entropy:.3f} · {case.responsable}",fontsize=10)
        ax.set_xlabel("SHAP · signo + aumenta riesgo",fontsize=9)
        ax.locator_params(axis="x",nbins=4)
    for ax in list(axes.flat)[len(cases):]:
        ax.set_visible(False)
    fig.suptitle(f"Nueve casos · {best_name} · {units}",y=1.01)
    savefig("10_shap_casos.png")
    ''')
    md("""
    ### Actividad 5.3 — Interpretación de cada caso

    En cada registro respondemos qué variables influyen, si el signo es coherente
    con la salida del modelo, para qué puede servir la explicación y qué confianza
    permite. «Coherente» significa consistencia interna, no una regla causal sobre
    una persona. Una explicación de una predicción errónea puede ser matemáticamente
    coherente y aun así exigir revisión. El simulador de humanos no utiliza SHAP;
    por tanto, no atribuimos a SHAP el incremento de F1.
    """)
    code('''
    for case in cases.itertuples():
        outcome = "coincide con la etiqueta" if case.y_true==case.y_ai else "es incorrecta frente a la etiqueta"
        signed_sum = float(aggregated.iloc[case.pos_test].sum())
        display(Markdown(f"**Fila {case.row_id} — {case.grupo}.** "
            f"**Variables:** {case.impactos}. "
            f"**Coherencia:** la suma de contribuciones es {signed_sum:+.3f} en {units}; "
            f"junto al valor base reconstruye la salida (control de aditividad superado), pero la predicción {outcome}. "
            f"**Utilidad:** el analista puede contrastar estos atributos con documentos del solicitante; "
            f"el responsable asignado es {case.responsable}. "
            f"**Confianza o confusión:** H={case.entropy:.3f} describe incertidumbre del modelo, "
            f"no validez causal. Los códigos categóricos y atributos correlacionados requieren consultar "
            f"el diccionario; ni SHAP ni baja entropía garantizan una decisión correcta o equitativa."))
    ''')
    md("""
    ## Parte 6 — Cálculo del puntaje final HACS (10%)

    ### Actividad 6.1 — Resultado de la optimización

    La búsqueda se ejecutó antes de prueba y se conserva íntegra en
    `busqueda_umbrales_validacion.csv`. Los umbrales no se retocan después de ver prueba,
    incluso si otro modelo o semilla ofrece un resultado mayor. El modelo defendido
    sigue siendo el elegido por F1 de validación.

    ### Actividad 6.2 — Tabla final, defensa y discusión

    La tabla de seis columnas replica la imagen solicitada. «Mejora» es diferencia
    absoluta de F1; la tabla ampliada incorpora puntos porcentuales, réplicas y costes.
    La condición del ranking se verifica con **F1 colaborativo > F1 IA** en la réplica
    reportada; no se oculta un incumplimiento. Completar los nombres del equipo.
    """)
    code('''
    ranking_cols = ["Grupo","Modelo","F1 - Sólo IA","F1 - IA + Human","Mejora","HACS"]
    display(results[ranking_cols].round(4))
    results[ranking_cols].to_csv(OUT / "tabla_entrega.csv",index=False)
    r = selected_result
    st = uncertainty_stats[-1]
    eligible_text = "supera" if r["Elegible_prueba"] else "NO supera"
    discussion = f"""### Resultado observado del modelo defendido: {best_name}

    Se eligió por F1 de validación ({validation.iloc[0].F1:.4f}), sin usar prueba.
    En prueba, F1 IA = **{r['F1 - Sólo IA']:.4f}**, F1 Humano–IA = **{r['F1 - IA + Human']:.4f}**,
    ΔF1 = **{r['Mejora']:+.4f}** ({r['Mejora_pp']:+.2f} puntos porcentuales) y HACS = **{r['HACS']:.4f}**.
    La realización reportada **{eligible_text}** la IA sola; elegible por mejora: **{r['Elegible_prueba']}**.

    La política A={chosen['A']:.4f}, B={chosen['B']:.4f} nats automatiza **{r['Coverage']:.1%}**:
    {r['N_IA']} IA, {r['N_Junior']} junior y {r['N_Senior']} senior, de 200 casos.
    Fue seleccionada por HACS con guardarraíles de mejora en validación; no se fuerza
    junior si la búsqueda prefiere A=B. Coste relativo IA **{r['Coste_IA']}** frente a
    colaborativo **{r['Coste_colab']}** (no incluye coste laboral).

    En 500 simulaciones, F1 colaborativo medio = **{r['F1_colab_MC_mean']:.4f}**,
    intervalo Monte Carlo central 95% = [{r['F1_colab_MC_p025']:.4f}, {r['F1_colab_MC_p975']:.4f}],
    y fracción de réplicas con mejora = **{r['Prob_mejora_MC']:.1%}**.
    El intervalo bootstrap pareado de ΔF1 en réplica 0 es [{ci[0]:.4f}, {ci[1]:.4f}].
    Si contiene cero, la mejora de esa muestra tiene incertidumbre material.

    Entropía media de aciertos **{st['mean_correct']:.4f}**, de errores **{st['mean_error']:.4f}**;
    AUC de detección de errores **{st['error_detection_auc']:.4f}** y p exploratorio
    Mann–Whitney **{st['mann_whitney_p']:.4g}**. La utilidad de entropía depende de estos
    valores: no asumimos que toda equivocación sea incierta ni que toda certeza acierte.

    **Appropriate reliance:** delegar según incertidumbre busca combinar automatización
    con revisión; el junior puede empeorar una decisión y un senior también puede fallar.
    Los beneficios observados dependen de las accuracies humanas **supuestas**, no son
    evidencia de rendimiento humano real. No hemos medido calibración completa,
    fatiga, dependencia humano–IA, tiempos ni equidad por subgrupos. El dataset es
    histórico y pequeño; una implementación real requiere validación externa y estudio
    humano antes de sostener beneficios operativos. SHAP hace la salida inspeccionable,
    pero no demuestra que la asistencia explicativa causó la mejora.
    """
    display(Markdown(discussion))
    (OUT / "discusion_resultados.md").write_text(discussion,encoding="utf-8")
    ''')
    md("""
    ### Comprobaciones finales y artefactos

    Reconciliamos fórmulas desde conteos y decisiones, y registramos semilla, modelo,
    política, SHA-256 del archivo, versiones y estado final. Se guarda el preprocesador
    junto al modelo para volver a predecir sin ajustarlo otra vez. Los archivos joblib
    deben abrirse únicamente si se confía en su origen.
    """)
    code('''
    for row in results.to_dict("records"):
        d = all_decisions[all_decisions.Modelo==row["Modelo"]]
        assert len(d)==200 and d.row_id.nunique()==200
        assert row["N_IA"]+row["N_Junior"]+row["N_Senior"]==200
        f_ai = f1_score(d.y_true,d.y_ai)
        f_final = f1_score(d.y_true,d.y_final)
        assert np.isclose(row["F1 - Sólo IA"],f_ai)
        assert np.isclose(row["F1 - IA + Human"],f_final)
        assert np.isclose(row["Mejora"],f_final-f_ai)
        assert np.isclose(row["HACS"],0.6*f_final+0.2*row["Coverage"]+0.2*row["Efficiency"])
    joblib.dump({"preprocessor":pre,"model":best_model,"name":best_name,"policy":chosen,
                 "features":FEATURES,"positive_class":"bad_credit"},OUT / "modelo_seleccionado.joblib")
    manifest = [{"name":p.name,"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in {"manifest.json","run_summary.json","__notebook__.ipynb"}]
    (OUT / "manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    checkpoint("done",status="complete",selected_model=best_name,shap_additivity_max_error=additivity_error,
               selected_test={k:(v.item() if isinstance(v,np.generic) else v) for k,v in selected_result.to_dict().items()},
               artifacts=[x["name"] for x in manifest])
    print("Estado:",summary["status"],"| Etapa:",summary["stage"],"| Modelo:",best_name)
    print("Artefactos:",len(manifest),"| Directorio:",OUT)
    ''')
    for i, cell in enumerate(cells):
        cell.id = f"hack11-{i:03d}"
    nb = nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                                                "language_info":{"name":"python","version":"3.11"}})
    nbf.validate(nb)
    path = ROOT / ("kaggle/v1/input/hack11_v1.ipynb" if mode=="kaggle" else "local/hack11_local_v1.ipynb")
    path.parent.mkdir(parents=True,exist_ok=True)
    nbf.write(nb,path)
    print(path, len(cells), "cells")


if __name__ == "__main__":
    build("kaggle")
    build("local")
    meta = {"id":"jeffreyamc/hack11-human-ai-credit-v1","title":"Hack11 Human AI Credit v1",
            "code_file":"hack11_v1.ipynb","language":"python","kernel_type":"notebook","is_private":True,
            "enable_gpu":True,"enable_internet":True,"machine_shape":"NvidiaTeslaT4",
            "dataset_sources":["jeffreyamc/hack11-german-credit-s13"],"competition_sources":[],"kernel_sources":[],"model_sources":[]}
    (ROOT / "kaggle/v1/input/kernel-metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
