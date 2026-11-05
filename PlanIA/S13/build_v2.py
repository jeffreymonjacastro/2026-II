"""Build the Kaggle v2 notebook; keep v1 and local v1 untouched."""
from pathlib import Path
import json
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
BASE = nbf.read(ROOT / "kaggle/v1/input/hack11_v1.ipynb", 4)
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))


def adapted(index):
    return (BASE.cells[index].source.replace("p_bad", "p_good")
            .replace("bad_credit", "good_credit").replace("mal crédito", "buen crédito")
            .replace("riesgo", "buen crédito"))


md(r"""
# Hackatón 11: ¿Confiar, verificar o escalar? — Kaggle v2

**Equipo:** completar nombres. **Clase positiva:** buen crédito (`1 = bueno`,
`0 = malo`). **Objetivo:** maximizar HACS, con mejora estricta
$\Delta F1=F1_{colaborativo}-F1_{IA}>0.01$, más de un punto porcentual.
También se informa mejora relativa, sin confundirla con puntos porcentuales.

Comparamos cinco familias y elegimos **tres finalistas por HACS de validación**.
Se ajustan hiperparámetros y umbral de clasificación sólo con entrenamiento;
la política y la selección final usan validación. Prueba no decide nada.
Los humanos son simulados, junior 70% y senior 95%, exactamente como v1.

Se conservan las seis partes del PDF. La prueba de 200 casos ya se inspeccionó en
v1: esta v2 no es un holdout externo nuevo. Cambiar la clase positiva cambia el
valor de F1; comparar con v1 requiere recodificar sus predicciones guardadas.

**Fuente:** Hofmann (1994), [UCI Statlog German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data),
[DOI 10.24432/C5NC77](https://doi.org/10.24432/C5NC77), CC BY 4.0.
Guía: `Sem13_Hackaton11.pdf` y diccionario `data/readme.md`.
""")
md("""
### Entorno y reproducibilidad

Kaggle proporciona los cinco clasificadores y SHAP. La sesión solicita Tesla T4;
el entrenamiento usa CPU, suficiente para 1,000 filas y reproducible. Se registran
versiones, hardware, hash del archivo, particiones, CV, políticas y resultados.
""")
code(BASE.cells[2].source + "\nfrom sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict\nfrom sklearn.base import clone")
md("""
### Configuración del experimento

La semilla de prueba y la réplica reportada se fijan antes de entrenar. No se
buscan semillas que favorezcan la mejora. Guardamos 500 simulaciones de validación
y 1,000 de prueba; el intervalo Monte Carlo refleja únicamente incertidumbre humana.
""")
code('''
MODE, SEED = "kaggle", 42
N_VAL_SIM, N_TEST_SIM = 500, 1000
JUNIOR_ACCURACY, SENIOR_ACCURACY = 0.70, 0.95
MIN_GAIN = 0.01
GROUP = "Equipo por completar"
DATA = next(p.parent for p in Path("/kaggle/input").rglob("german.data"))
OUT = Path("/kaggle/working")
OUT.mkdir(parents=True,exist_ok=True)
versions = {p:metadata.version(p) for p in ["numpy","pandas","scikit-learn","scipy","matplotlib","shap","xgboost","lightgbm","catboost"]}
summary = {"status":"running","stage":"setup","experiment":"v2","positive_class":"good_credit",
           "label_mapping":{"0":"bad_credit","1":"good_credit"},"seed":SEED,
           "python":sys.version,"versions":versions,"human_simulation_only":True,
           "minimum_absolute_f1_gain_strict":MIN_GAIN,"test_seen_in_v1":True}
def checkpoint(stage,**values):
    summary.update(stage=stage,**values)
    (OUT/"run_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT/name,dpi=150,bbox_inches="tight")
    plt.show()
    plt.close()
gpu = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],capture_output=True,text=True).stdout.strip()
checkpoint("setup",hardware=gpu,estimator_device="CPU")
print("v2 · positivo = buen crédito | Hardware:",gpu)
''')
md("""
## Parte 1 — Comprensión del problema y exploración (10%)

### Actividad 1.1 — Contexto y coste de equivocarse

Aprobar a un cliente malo puede causar pérdida de capital; rechazar a un buen
cliente pierde negocio y puede perjudicarlo injustamente. El README asigna coste
cinco al primer error y uno al segundo. Con **positivo = bueno**, FP significa
aprobar a un cliente malo y FN rechazar a uno bueno: **coste = 5 FP + FN**.

Medimos F1 binario de **buen crédito**, según la solicitud de esta v2. F1 de mal
crédito y F1 macro quedan como métricas complementarias; ni el cambio de etiqueta
ni un HACS mayor demuestran por sí solos menor riesgo financiero o mayor equidad.

### Actividad 1.2 — Datos y diccionario

Usamos `german.data`, 20 atributos y 1,000 filas. La representación numérica
alternativa no añade observaciones. A65/A124 son categorías del diccionario,
no celdas nulas. Conservamos los atributos personales para reproducir el ejercicio;
no asumimos que ello permita un despliegue bancario.
""")
source = adapted(6).replace("df.original_class == 2", "df.original_class == 1")
source = source.replace("[700, 300]", "[300, 700]").replace("class_0_good", "class_0_bad").replace("class_1_bad", "class_1_good")
code(source)
md(BASE.cells[7].source.replace("la etiqueta binaria", "la etiqueta de buen crédito"))
code(adapted(8).replace('["Bueno", "Malo"]', '["Malo", "Bueno"]'))
md("""
## Parte 2 — Desarrollo del modelo base (20%)

### Actividad 2.1 — Particiones y preprocesamiento

Conservamos exactamente las filas de v1: 600 entrenamiento, 200 validación y
200 prueba. Estratificamos usando `1-y`, la codificación de v1, para reproducir
sus índices después de invertir la clase positiva. La proporción es 70% buenos.

Imputación preventiva, one-hot y escalamiento se ajustan sólo con entrenamiento.
Durante CV, el **pipeline completo** se ajusta de nuevo dentro de cada fold;
no preprocesamos los folds con estadísticas del fold que se evalúa.
""")
source = BASE.cells[10].source.replace("stratify=y,", "stratify=1-y,").replace("stratify=y[idx_trainval]", "stratify=1-y[idx_trainval]")
source = source.replace('{0: "Bueno", 1: "Malo"}', '{0: "Malo", 1: "Bueno"}')
code(source)
md("""
### Actividad 2.2 — Candidatos y búsqueda de hiperparámetros

Evaluamos XGBoost, LightGBM, CatBoost, Random Forest y Logistic Regression.
GridSearchCV usa cinco folds estratificados **sólo en entrenamiento**, puntuando
F1 de buen crédito. Cada familia aporta una configuración final. No se aplica
SMOTE, ni se cambian las accuracies humanas. Las curvas ROC usan P(buen crédito).
""")
code('''
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
candidates = {
    "XGBoost": (XGBClassifier(learning_rate=0.05,subsample=0.9,colsample_bytree=0.9,
        reg_lambda=3,eval_metric="logloss",random_state=SEED,n_jobs=2),
        {"model__n_estimators":[120,260],"model__max_depth":[2,3]}),
    "LightGBM": (LGBMClassifier(learning_rate=0.05,min_child_samples=20,reg_lambda=3,
        verbosity=-1,random_state=SEED,n_jobs=2),
        {"model__n_estimators":[120,260],"model__num_leaves":[5,9]}),
    "CatBoost": (CatBoostClassifier(learning_rate=0.05,l2_leaf_reg=5,loss_function="Logloss",
        random_seed=SEED,verbose=False,allow_writing_files=False,thread_count=2),
        {"model__iterations":[150,300],"model__depth":[3,5]}),
    "Random Forest": (RandomForestClassifier(n_estimators=250,random_state=SEED,n_jobs=2),
        {"model__max_depth":[5,8,None],"model__min_samples_leaf":[2,5]}),
    "Logistic Regression": (LogisticRegression(max_iter=2000,solver="lbfgs",random_state=SEED),
        {"model__C":[0.05,0.2,1,5],"model__class_weight":[None,"balanced"]})}
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=2026)
cv_folds = np.full(len(idx_train),-1,dtype=int)
for fold,(_,held) in enumerate(cv.split(X.iloc[idx_train],yt)):
    cv_folds[held] = fold
pd.DataFrame({"row_id":idx_train,"fold":cv_folds,"y_good":yt}).to_csv(OUT/"folds_entrenamiento.csv",index=False)
''')
md("""
### Métricas, entropía y simulación: funciones auditables

F1 principal = `2 TP / (2 TP + FP + FN)`, positivo bueno. La entropía se calcula
con logaritmo natural y alcanza ln(2) en p=0.5. El enrutador sólo ve entropía;
las etiquetas reales se usan exclusivamente para evaluar y simular respuestas.
Junior/senior tienen errores Bernoulli, como v1, con generadores continuos por caso.
""")
metrics_source = BASE.cells[14].source.split("val_prob, test_prob")[0]
metrics_source = metrics_source.replace("fp + 5*fn", "5*fp + fn").replace("fp+5*fn", "5*fp+fn")
metrics_source = metrics_source.replace('"F1_macro":', '"F1_bad": f1_score(truth,pred,pos_label=0,zero_division=0),\n            "F1_macro":')
code(metrics_source + "\n" + BASE.cells[16].source.split("pv =")[0] + "\n" + BASE.cells[20].source.split("jval,sval =")[0])
md("""
### Ajuste y umbral binario de cada familia

Tras seleccionar hiperparámetros con CV, calculamos probabilidades out-of-fold
de entrenamiento para elegir el umbral que maximiza F1 de **IA sola**. Esto
fortalece el baseline y evita fabricar mejora usando un umbral arbitrariamente
malo. La puntuación OOF participa en el ajuste y no es la evaluación final.
Guardamos los folds y las probabilidades para auditarlo.
""")
code('''
fitted,models,val_prob,test_prob,thresholds_cls = {},{},{},{},{}
training_rows,cv_tables,oof_rows = [],[],[]
for name,(estimator,grid) in candidates.items():
    start = time.perf_counter()
    pipe = Pipeline([("pre",clone(pre)),("model",estimator)])
    search_cv = GridSearchCV(pipe,grid,scoring="f1",cv=cv,refit=True,n_jobs=1,error_score="raise")
    search_cv.fit(X.iloc[idx_train],yt)
    fitted[name] = search_cv.best_estimator_
    models[name] = fitted[name].named_steps["model"]
    oof = cross_val_predict(clone(fitted[name]),X.iloc[idx_train],yt,cv=cv,method="predict_proba",n_jobs=1)[:,1]
    threshold_grid = np.unique(np.r_[np.linspace(0.1,0.9,81),0.5])
    threshold_f1 = np.array([f1_score(yt,oof>=t) for t in threshold_grid])
    # Ante un empate, el umbral más cercano a 0.5 evita extremos innecesarios.
    optimum = threshold_f1.max()
    tied = threshold_grid[np.isclose(threshold_f1,optimum,rtol=0,atol=1e-12)]
    tau = float(tied[np.argmin(abs(tied-0.5))])
    thresholds_cls[name] = tau
    val_prob[name] = fitted[name].predict_proba(X.iloc[idx_val])[:,1]
    training_rows.append({"Modelo":name,"Threshold":tau,"CV_F1_good":float(search_cv.best_score_),
        "OOF_F1_good":float(optimum),**metrics(yv,val_prob[name]>=tau,val_prob[name]),
        "Brier":brier_score_loss(yv,val_prob[name]),"Segundos":time.perf_counter()-start,
        "Parametros":json.dumps(search_cv.best_params_,sort_keys=True)})
    table = pd.DataFrame(search_cv.cv_results_)
    table["Modelo"] = name
    cv_tables.append(table[["Modelo","params","mean_test_score","std_test_score","rank_test_score"]])
    oof_rows.append(pd.DataFrame({"Modelo":name,"row_id":idx_train,"fold":cv_folds,"y_good":yt,"p_good_oof":oof}))
    print(name,"| CV F1:",round(search_cv.best_score_,4),"| umbral:",tau)
    pd.DataFrame(training_rows).to_csv(OUT/"modelos_validacion.csv",index=False)
    checkpoint("training",families_finished=len(training_rows))
validation = pd.DataFrame(training_rows)
pd.concat(cv_tables,ignore_index=True).to_csv(OUT/"grid_search_entrenamiento.csv",index=False)
pd.concat(oof_rows,ignore_index=True).to_csv(OUT/"probabilidades_oof_entrenamiento.csv",index=False)
display(validation[["Modelo","Threshold","CV_F1_good","F1","F1_bad","ROC_AUC"]].round(4))
''')
md(r"""
### Selección conjunta de tres modelos y políticas — validación

Para cada familia, probamos A/B en cuantiles de entropía de validación.
$H<A$: IA; $A\leq H<B$: junior; $H\geq B$: senior. Permitir A=B evita forzar
junior cuando no aporta: HACS penaliza igual una revisión junior que una senior.

Elegimos máximo HACS medio entre políticas con **mejora media ≥0.02** y
**percentil 10 de mejora >0.01**, en 500 réplicas. Este margen busca que el requisito
estricto >0.01 no dependa de una respuesta humana afortunada; no lo garantiza
en prueba. Después seleccionamos las tres familias elegibles con mayor HACS.
Se guardan todas las políticas y el orden antes de calcular predicciones de prueba.
""")
code('''
jval,sval = simulate_analysts(yv,N_VAL_SIM,seed=1001)
policies,grid_rows = {},[]
for name,prob in val_prob.items():
    u,ai = entropy(prob),(prob>=thresholds_cls[name]).astype(int)
    baseline = f1_score(yv,ai)
    cuts = np.unique(np.r_[0,np.quantile(u,np.linspace(0.05,0.995,32)),np.log(2)+1e-8])
    rows = []
    for a in cuts:
        for b in cuts[cuts>=a]:
            makers = route(u,a,b)
            fs = f1_many(yv,collaborate(ai,makers,jval,sval))
            delta = fs-baseline
            coverage = float(np.mean(makers=="IA"))
            row = {"Modelo":name,"Threshold":thresholds_cls[name],"A":float(a),"B":float(b),
                "F1_IA":float(baseline),"F1_colab_mean":float(fs.mean()),"Mejora":float(delta.mean()),
                "Delta_p10":float(np.quantile(delta,0.1)),"Prob_mejora_mayor_1pp":float(np.mean(delta>MIN_GAIN)),
                "Coverage":coverage,"Efficiency":coverage,"HACS":float(.6*fs.mean()+.4*coverage),
                "Junior":int(sum(makers=="Junior")),"Senior":int(sum(makers=="Senior"))}
            row["Elegible_validacion"] = row["Mejora"]>=0.02 and row["Delta_p10"]>MIN_GAIN
            rows.append(row)
    grid = pd.DataFrame(rows)
    feasible = grid[grid.Elegible_validacion]
    choice = (feasible if len(feasible) else grid).sort_values(["HACS","Coverage"],ascending=False).iloc[0].to_dict()
    policies[name] = choice
    grid_rows.extend(rows)
search = pd.DataFrame(grid_rows)
search.to_csv(OUT/"busqueda_umbrales_validacion.csv",index=False)
policy_table = pd.DataFrame(policies.values()).sort_values(["Elegible_validacion","HACS"],ascending=False)
eligible = policy_table[policy_table.Elegible_validacion]
assert len(eligible)>=3,"No hay tres familias elegibles; detener antes de evaluar prueba."
finalist_names = eligible.head(3).Modelo.tolist()
best_name = finalist_names[0]
best_model = models[best_name]
chosen = policies[best_name]
policy_table["Finalista"] = policy_table.Modelo.isin(finalist_names)
policy_table.to_csv(OUT/"seleccion_modelos_validacion.csv",index=False)
policy_table.to_csv(OUT/"politicas_congeladas.csv",index=False)
(OUT/"politicas_congeladas.json").write_text(json.dumps(policies,indent=2),encoding="utf-8")
(OUT/"finalistas_congelados.json").write_text(json.dumps({"selection_split":"validation","criterion":"mean_HACS",
    "finalists":finalist_names,"selected":best_name,"classification_thresholds":thresholds_cls},indent=2),encoding="utf-8")
pre = fitted[best_name].named_steps["pre"]
Ztrain,Zval,Ztest = pre.transform(X.iloc[idx_train]),pre.transform(X.iloc[idx_val]),pre.transform(X.iloc[idx_test])
names = pre.get_feature_names_out()
for name in finalist_names:
    assert np.array_equal(fitted[name].named_steps["pre"].get_feature_names_out(),names)
    assert np.allclose(fitted[name].named_steps["pre"].transform(X.iloc[idx_train]),Ztrain)
checkpoint("finalists_frozen",selected_model=best_name,finalists=finalist_names,policy=chosen,
           validation_repetitions=N_VAL_SIM)
display(policy_table[["Modelo","F1_IA","F1_colab_mean","Mejora","Delta_p10","Coverage","HACS","Finalista"]].round(4))
print("Tres finalistas congelados:",finalist_names,"| Defendido:",best_name)
''')
md(r"""
## Parte 3 — Cuantificación de incertidumbre (20%)

### Actividades 3.1 y 3.2 — Probabilidades y entropía

Usamos $p=P(buen\ crédito\mid x)$ y $H=-p\ln(p)-(1-p)\ln(1-p)$, en nats.
El máximo sigue siendo ln(2), aunque el umbral binario aprendido sea distinto
de 0.5. Entropía y umbral de clasificación son conceptos diferentes. La primera
describe la distribución predictiva; el segundo convierte probabilidades en clase.
""")
source = adapted(16).split("pv =")[1]
source = "pv =" + source
source = source.replace("(pv>=0.5)", "(pv>=thresholds_cls[best_name])")
code(source)
md(BASE.cells[17].source)
source = adapted(18).replace("(prob>=0.5)", "(prob>=thresholds_cls[best_name])")
code(source)
md("""
## Parte 4 — Diseño del sistema Humano–IA (25%)

### Actividad 4.1 — Umbrales y evidencia experimental

Los umbrales seleccionados se muestran junto a las candidatas: automatización
frente a HACS. Se justifica la política con validación y variabilidad humana,
sin revisarla después de mirar prueba. Los modelos finalistas también se eligen
en validación. La prueba de v1 se reutiliza para comparación, no como test nuevo.

### Actividad 4.2 — Junior y senior simulados

Junior acierta con probabilidad 0.70 y senior 0.95. «Acierto» significa accuracy,
no Precision. Cuando fallan, invierten la clase. Se reutilizan respuestas dentro
de cada split para comparar políticas de forma pareada, sin reiniciar la semilla
por cliente. El simulador usa etiquetas para generar respuestas sintéticas;
**el enrutador nunca recibe esas etiquetas**. No modelamos fatiga ni errores
correlacionados con dificultad; no hay evidencia de analistas humanos reales.

### Actividades 4.3 y 4.4 — Pipeline y comparación

El pipeline produce probabilidad, entropía, responsable y decisión final.
Cada caso tiene exactamente un responsable. La evaluación incluirá IA sola,
junior para todos, senior para todos y colaboración. A=B implica cero juniors;
es admisible porque el score no distingue coste laboral según experiencia.
""")
code('''
display(policy_table[["Modelo","Threshold","A","B","Junior","Senior","Coverage","HACS","Finalista"]].round(4))
selected_grid = search[search.Modelo==best_name]
fig,ax = plt.subplots(figsize=(9,4.5))
ax.scatter(selected_grid.Coverage,selected_grid.HACS,s=15,alpha=0.4,color=COLORS["IA"],label="Candidatas")
ax.scatter([chosen["Coverage"]],[chosen["HACS"]],s=95,marker="*",color=COLORS["Humano–IA"],label="Elegida")
ax.set(xlabel="Cobertura IA",ylabel="HACS medio",title=f"Políticas en validación · {best_name}",xlim=(0,1),ylim=(0,1))
ax.legend()
savefig("04_politicas_validacion.png")
''')
md("""
### Evaluación de prueba — tres finalistas congelados

Se reporta la réplica 0, con semilla fijada, y 1,000 réplicas nuevas. La mejora
estricta >0.01 se comprueba en la realización y en la media Monte Carlo. Si un
modelo falla el requisito, se etiqueta como incumplimiento; no se cambia por un
cuarto modelo mirando prueba. La tabla de cada finalista incluye F1 de ambas clases,
coste, cobertura, humanos usados y fracción de réplicas con mejora >1 punto.
""")
code('''
jtest,stest = simulate_analysts(ye,N_TEST_SIM,seed=20261104)
final_rows,sim_rows,decision_frames,base_rows = [],[],[],[]
for name in finalist_names:
    prob = fitted[name].predict_proba(X.iloc[idx_test])[:,1]
    test_prob[name] = prob
    ai,u = (prob>=thresholds_cls[name]).astype(int),entropy(prob)
    policy = policies[name]
    makers = route(u,policy["A"],policy["B"])
    decisions = collaborate(ai,makers,jtest,stest).astype(int)
    coverage = float(np.mean(makers=="IA"))
    efficiency = float(1-np.mean(makers!="IA"))
    fm = f1_many(ye,decisions)
    base,point = metrics(ye,ai,prob),metrics(ye,decisions[0])
    base_rows.append({"Modelo":name,"Threshold":thresholds_cls[name],**base,"Brier":brier_score_loss(ye,prob)})
    delta = point["F1"]-base["F1"]
    row = {"Grupo":GROUP,"Modelo":name,"F1 - Sólo IA":base["F1"],"F1 - IA + Human":point["F1"],
        "Mejora":delta,"Mejora_pp":100*delta,"Mejora_relativa_pct":100*delta/base["F1"],
        "HACS":.6*point["F1"]+.2*coverage+.2*efficiency,"Coverage":coverage,"Efficiency":efficiency,
        "N_IA":int(sum(makers=="IA")),"N_Junior":int(sum(makers=="Junior")),"N_Senior":int(sum(makers=="Senior")),
        "F1_bad_IA":base["F1_bad"],"F1_bad_colab":point["F1_bad"],
        "F1_colab_MC_mean":float(fm.mean()),"F1_colab_MC_p025":float(np.quantile(fm,.025)),
        "F1_colab_MC_p975":float(np.quantile(fm,.975)),"Mejora_MC_mean":float(fm.mean()-base["F1"]),
        "Prob_mejora_mayor_1pp_MC":float(np.mean(fm-base["F1"]>MIN_GAIN)),
        "HACS_MC_mean":float(.6*fm.mean()+.4*coverage),"Coste_IA":base["Coste"],"Coste_colab":point["Coste"],
        "Cumple_mayor_1pp":bool(delta>MIN_GAIN),"Cumple_MC_mayor_1pp":bool(fm.mean()-base["F1"]>MIN_GAIN),
        "Seleccionado_validacion":name==best_name}
    final_rows.append(row)
    sim_rows.extend({"Modelo":name,"replica":r,"F1":float(f),"Mejora":float(f-base["F1"]),
        "HACS":float(.6*f+.4*coverage)} for r,f in enumerate(fm))
    decision_frames.append(pd.DataFrame({"Modelo":name,"row_id":idx_test,"y_true":ye,"p_good":prob,
        "threshold":thresholds_cls[name],"entropy":u,"y_ai":ai,"responsable":makers,
        "y_junior":jtest[0],"y_senior":stest[0],"y_final":decisions[0]}))
results = pd.DataFrame(final_rows)
all_decisions = pd.concat(decision_frames,ignore_index=True)
all_decisions.to_csv(OUT/"decisiones_prueba.csv",index=False)
results.to_csv(OUT/"tabla_resultados.csv",index=False)
pd.DataFrame(sim_rows).to_csv(OUT/"simulaciones_prueba.csv",index=False)
pd.DataFrame(base_rows).to_csv(OUT/"modelos_prueba.csv",index=False)
display(results[["Modelo","F1 - Sólo IA","F1 - IA + Human","Mejora_pp","HACS","Coverage","Cumple_mayor_1pp"]].round(4))
display(results[["Modelo","F1_colab_MC_mean","Mejora_MC_mean","HACS_MC_mean","Prob_mejora_mayor_1pp_MC"]].round(4))
selected_decisions = all_decisions[all_decisions.Modelo==best_name].reset_index(drop=True)
selected_result = results[results.Modelo==best_name].iloc[0]
comparison = pd.DataFrame([
    {"Sistema":"IA",**metrics(ye,selected_decisions.y_ai,test_prob[best_name])},
    {"Sistema":"Junior",**metrics(ye,jtest[0])},
    {"Sistema":"Senior",**metrics(ye,stest[0])},
    {"Sistema":"Humano–IA",**metrics(ye,selected_decisions.y_final)}])
comparison.to_csv(OUT/"comparacion_sistemas.csv",index=False)
display(comparison[["Sistema","Accuracy","Precision","Recall","F1","F1_bad","Coste"]].round(4))
''')
md(BASE.cells[25].source.replace("modelo IA", "modelo IA, positivo = buen crédito"))
code(adapted(26).replace('["Bueno","Malo"]', '["Malo","Bueno"]'))
md(BASE.cells[27].source.replace("500", "1,000"))
code(BASE.cells[28].source.replace("500 réplicas", "1,000 réplicas"))
md(adapted(29))
code(adapted(30))
md(BASE.cells[31].source)
code(adapted(32))
md(adapted(33))
code(adapted(34))
md(r"""
## Parte 6 — Puntaje final HACS (10%)

### Actividad 6.1 — Optimización y requisito estricto

Se seleccionaron modelo y política por HACS de validación, con margen preventivo
de mejora. Las decisiones se congelaron antes de prueba. Para el requisito del
usuario se comprueba $\Delta F1>0.01$; una diferencia exactamente 0.01 no alcanza.

$Coverage=N_{IA}/N$, $Efficiency=1-N_{humanos}/N$ y
$HACS=0.6F1_{colab}+0.2Coverage+0.2Efficiency$.

Cobertura y eficiencia coinciden. El HACS se mantiene idéntico al PDF; no se
añaden bonificaciones por superar 0.9 ni se penaliza distinto al senior.

### Actividad 6.2 — Tabla de entrega y defensa

«Mejora» es diferencia absoluta de F1; «Mejora_pp» la expresa en puntos
porcentuales. La comparación histórica debe usar la misma clase positiva:
un HACS mayor tras invertir la etiqueta no demuestra por sí solo un modelo mejor.
""")
code('''
ranking_cols = ["Grupo","Modelo","F1 - Sólo IA","F1 - IA + Human","Mejora","HACS"]
results[ranking_cols].to_csv(OUT/"tabla_entrega.csv",index=False)
display(results[ranking_cols].round(4))
r = selected_result
st = uncertainty_stats[-1]
val_selected = validation[validation.Modelo==best_name].iloc[0]
discussion = f"""### Modelo defendido: {best_name}

Elegido por HACS medio de validación **{chosen['HACS']:.4f}**, entre cinco familias.
Umbral de clasificación aprendido con OOF = **{thresholds_cls[best_name]:.3f}**;
política de entropía A={chosen['A']:.4f}, B={chosen['B']:.4f} nats.
En validación, mejora media={chosen['Mejora']:.4f} y percentil 10 de mejora={chosen['Delta_p10']:.4f}.

En prueba: F1 de buen crédito IA **{r['F1 - Sólo IA']:.4f}**, colaborativo
**{r['F1 - IA + Human']:.4f}**, mejora **{r['Mejora_pp']:+.2f} puntos porcentuales**
({r['Mejora_relativa_pct']:+.2f}% relativo) y HACS **{r['HACS']:.4f}**.
Cumple mejora estricta >0.01 en réplica reportada: **{r['Cumple_mayor_1pp']}**;
en media Monte Carlo: **{r['Cumple_MC_mayor_1pp']}**.

Decisiones: {r['N_IA']} IA, {r['N_Junior']} junior y {r['N_Senior']} senior;
cobertura={r['Coverage']:.1%}. Coste original 5 FP + FN: IA **{r['Coste_IA']}**,
colaborativo **{r['Coste_colab']}**. F1 de mal crédito: IA {r['F1_bad_IA']:.4f},
colaborativo {r['F1_bad_colab']:.4f}; no se oculta el comportamiento de la otra clase.

En 1,000 réplicas: F1 colaborativo medio **{r['F1_colab_MC_mean']:.4f}**,
intervalo Monte Carlo central 95% [{r['F1_colab_MC_p025']:.4f},{r['F1_colab_MC_p975']:.4f}],
HACS medio **{r['HACS_MC_mean']:.4f}** y mejora >1 punto en **{r['Prob_mejora_mayor_1pp_MC']:.1%}**
de réplicas. IC bootstrap pareado de mejora, réplica 0: [{ci[0]:.4f},{ci[1]:.4f}].

Entropía media: aciertos {st['mean_correct']:.4f}, errores {st['mean_error']:.4f};
AUC de detección de errores {st['error_detection_auc']:.4f}. No toda certeza acierta.

**Límites:** accuracies junior/senior son supuestos pedagógicos; no representan
evidencia humana real. SHAP explica el clasificador, no mejora al simulador ni
demuestra causalidad. No medimos fatiga, dependencia humano–IA, coste laboral ni
equidad por grupos. Los umbrales de colaboración se buscan sobre sólo 200 casos
de validación y pueden sobreajustar. Prueba ya se vio en v1; se necesita un
conjunto externo y un experimento humano para sostener beneficios operativos.
El cambio de F1 de malo a bueno altera la escala de comparación con v1.
"""
display(Markdown(discussion))
(OUT/"discusion_resultados.md").write_text(discussion,encoding="utf-8")
''')
md("""
### Auditoría de fórmulas, trazabilidad y cierre

Las verificaciones reconcilian decisiones, etiquetas, F1 positivo bueno, coste,
HACS y mejora. El estado registra por separado si los tres finalistas cumplen
el requisito >1 punto; una ejecución técnicamente completa no oculta un fallo
experimental. Se guarda el pipeline completo y el umbral aprendido por finalista.
""")
code('''
for row in results.to_dict("records"):
    d = all_decisions[all_decisions.Modelo==row["Modelo"]]
    assert len(d)==200 and d.row_id.nunique()==200
    assert np.array_equal(d.y_true,(df.iloc[d.row_id].original_class==1).astype(int))
    assert row["N_IA"]+row["N_Junior"]+row["N_Senior"]==200
    f_ai,f_final = f1_score(d.y_true,d.y_ai),f1_score(d.y_true,d.y_final)
    assert np.isclose(row["F1 - Sólo IA"],f_ai)
    assert np.isclose(row["F1 - IA + Human"],f_final)
    assert np.isclose(row["Mejora"],f_final-f_ai)
    assert row["Cumple_mayor_1pp"]==(f_final-f_ai>MIN_GAIN)
    assert np.isclose(row["HACS"],.6*f_final+.2*row["Coverage"]+.2*row["Efficiency"])
    joblib.dump({"pipeline":fitted[row["Modelo"]],"policy":policies[row["Modelo"]],
        "threshold":thresholds_cls[row["Modelo"]],"features":FEATURES,"positive_class":"good_credit"},
        OUT/(row["Modelo"].lower().replace(" ","_")+".joblib"))
manifest = [{"name":p.name,"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
    for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in {"manifest.json","run_summary.json","__notebook__.ipynb"}]
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
checkpoint("done",status="complete",selected_model=best_name,finalists=finalist_names,
    all_three_gain_gt_1pp=bool(results.Cumple_mayor_1pp.all()),
    all_three_mean_gain_gt_1pp=bool(results.Cumple_MC_mayor_1pp.all()),
    selected_test={k:(v.item() if isinstance(v,np.generic) else v) for k,v in r.to_dict().items()},
    shap_additivity_max_error=additivity_error,bootstrap_delta_ci=ci.tolist(),
    artifacts=[x["name"] for x in manifest])
print("Estado:",summary["status"],"| Etapa:",summary["stage"],"| Finalistas:",finalist_names)
print("Mejora >1pp, tres modelos:",summary["all_three_gain_gt_1pp"],"| Media MC:",summary["all_three_mean_gain_gt_1pp"])
''')

for i,cell in enumerate(cells):
    cell.id = f"hack11-v2-{i:03d}"
nb = nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                                             "language_info":{"name":"python"}})
nbf.validate(nb)
for c in cells:
    if c.cell_type=="code":
        compile(c.source,c.id,"exec")
folder = ROOT/"kaggle/v2/input"
folder.mkdir(parents=True,exist_ok=True)
nbf.write(nb,folder/"hack11_v2.ipynb")
meta = json.loads((ROOT/"kaggle/v1/input/kernel-metadata.json").read_text())
meta.update(id="jeffreyamc/hack11-human-ai-credit-v2-good-credit",title="Hack11 Human AI Credit v2 - Good Credit",code_file="hack11_v2.ipynb")
(folder/"kernel-metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
print("v2 generada:",len(cells),"celdas;",sum(c.cell_type=="code" for c in cells),"celdas de código")
