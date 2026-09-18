# Reporte de implementación y despliegue - Bank Marketing

Fecha de ejecución: 18 de septiembre de 2026  
Proyecto de Google Cloud: `plania-workshop`  
Servicio: `bank-marketing-api`  
Región: `us-central1`

## 1. Alcance y separación de instrucciones

El objetivo de la hackathon es:

1. Predecir si un cliente aceptará un depósito a plazo (`yes`/`no`).
2. Excluir obligatoriamente `duration` del modelo final por _data leakage_.
3. Explorar los datos, entrenar al menos dos modelos y evaluar F1 y balanced accuracy.
4. Seleccionar hiperparámetros, reentrenar con todos los datos y guardar el modelo.
5. Implementar `predict(input_dict)` con validación, predicción y probabilidades.
6. Exponer `POST /predict` con FastAPI.
7. Crear un cliente automático con al menos cinco ejemplos.
8. Desplegar en GCP y justificar el nivel de implementación y sus riesgos.

## 2. Archivos entregados

| Archivo                          | Propósito                                                            |
| -------------------------------- | -------------------------------------------------------------------- |
| `bank_marketing_model.ipynb`     | Notebook detallado, ejecutado y con resultados/gráficos guardados.   |
| `create_notebook.py`             | Generador reproducible del notebook con `nbformat`.                  |
| `model.joblib`                   | Tubería final serializada: preprocesamiento, modelo y umbral.        |
| `model_metadata.json`            | Métricas, variables, hiperparámetros y trazabilidad del modelo.      |
| `inference.py`                   | Función `predict(input_dict)` y validaciones independientes de HTTP. |
| `app.py`                         | API FastAPI con `/`, `/health` y `/predict`.                         |
| `client.py`                      | Cliente automático de cinco ejemplos.                                |
| `sample_request.json`            | Cuerpo JSON para la prueba manual del docente.                       |
| `Dockerfile`                     | Imagen reproducible para Cloud Run.                                  |
| `requirements.txt`               | Dependencias fijadas para el contenedor.                             |
| `deploy.ps1`                     | Despliegue reproducible mediante Google Cloud CLI.                   |
| `deploy.sh`                      | Despliegue equivalente para Bash, Linux, macOS o Git Bash.           |
| `.dockerignore`, `.gcloudignore` | Excluyen datos, notebooks y temporales de la imagen/subida.          |

## 3. Datos y control de leakage

Se usó exclusivamente `bank.csv` con 17 variables

- Filas: **4,521**
- Columnas originales: **17**
- Clase `yes`: **521 (11.52%)**
- Clase `no`: **4,000 (88.48%)**
- Nulos detectados: **0**
- Duplicados exactos: el notebook los contabiliza y conserva porque pueden representar contactos observados; no se usa información de identidad para deduplicar con seguridad.

`duration` se inspeccionó únicamente para explicar el riesgo: se conoce después de la interacción y cambia materialmente según el resultado. Después se eliminó antes de cualquier partición o entrenamiento. El artefacto serializado contiene 15 predictores y una comprobación automatizada confirmó que `duration` no figura en `feature_names`. La API también rechaza campos extra, por lo que un JSON que incluya `duration` recibe HTTP 422.

## 4. Metodología

1. Separación estratificada 60% entrenamiento, 20% validación y 20% test.
2. Preprocesamiento dentro de una `Pipeline`:
   - mediana y estandarización para variables numéricas;
   - moda y one-hot encoding para categóricas;
   - categorías nuevas ignoradas de forma segura en inferencia.
3. Validación cruzada estratificada de 5 folds solo en entrenamiento.
4. Comparación de regresión logística y Random Forest.
5. Selección por F1 promedio en CV.
6. Elección del umbral que maximiza F1 en validation.
7. Evaluación única en test.
8. Reentrenamiento del modelo ganador con las 4,521 filas y serialización de la tubería completa.

Este diseño evita usar test para seleccionar modelo, hiperparámetros o umbral. Las métricas de referencia proceden de test; no se presentan métricas optimistas calculadas sobre el conjunto completo reentrenado.

## 5. Resultados

### 5.1 Comparación en validación cruzada

| Modelo              |      F1 CV | Desv. F1 | Balanced accuracy CV | ROC-AUC CV |
| ------------------- | ---------: | -------: | -------------------: | ---------: |
| Random Forest       | **0.4062** |   0.0259 |               0.6642 | **0.7471** |
| Regresión logística |     0.3315 |   0.0177 |           **0.6693** |     0.7277 |

Se seleccionó **Random Forest** por tener el mayor F1 promedio, la métrica de selección declarada. Sus mejores hiperparámetros fueron `max_depth=10` y `min_samples_leaf=5`. El umbral ajustado en validation fue **0.4853**.

### 5.2 Métricas finales en test

| Métrica           |      Valor |
| ----------------- | ---------: |
| F1                | **0.3692** |
| Balanced accuracy | **0.6634** |
| ROC-AUC           | **0.7371** |
| Precision         |     0.3077 |
| Recall            |     0.4615 |

Matriz de confusión: 693 verdaderos negativos, 108 falsos positivos, 56 falsos negativos y 48 verdaderos positivos. El desempeño es moderado y coherente con el problema desbalanceado y con la exclusión correcta de `duration`; no debe interpretarse como causalidad ni como garantía comercial.

## 6. API y despliegue en Google Cloud

Se utilizó la biblioteca oficial `Google Cloud SDK` (`/websites/cloud_google_sdk`). El flujo adoptado usa `gcloud run deploy --source .`; la documentación oficial confirma que esta modalidad usa Cloud Build y Artifact Registry, y que un `Dockerfile` presente se usa para construir la imagen: [Deploy services from source code](https://cloud.google.com/run/docs/deploying-source-code).

Comandos encapsulados en `deploy.ps1`:

```powershell
gcloud config set project plania-workshop
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy bank-marketing-api `
  --source . `
  --region us-central1 `
  --allow-unauthenticated `
  --memory 1Gi `
  --cpu 1 `
  --min-instances 0 `
  --max-instances 2 `
  --quiet
```

Resultado verificado:

- Revisión: `bank-marketing-api-00001-jvx`
- Tráfico: **100%** a la revisión más reciente
- Recursos: **1 CPU**, **1 GiB**, máximo **2 instancias**
- URL pública: <https://bank-marketing-api-i457dzdera-uc.a.run.app>
- Swagger UI: <https://bank-marketing-api-i457dzdera-uc.a.run.app/docs>
- Health check: <https://bank-marketing-api-i457dzdera-uc.a.run.app/health>

## 7. Pruebas manuales

### Manual en Swagger

1. Abrir <https://bank-marketing-api-i457dzdera-uc.a.run.app/docs>.
2. Expandir `POST /predict`.
3. Pulsar **Try it out**.
4. Pegar el contenido de `sample_request.json`.
5. Pulsar **Execute** y comprobar HTTP 200.

### Manual con PowerShell

```powershell
$body = Get-Content -Raw .\sample_request.json
Invoke-RestMethod `
  -Uri "https://bank-marketing-api-i457dzdera-uc.a.run.app/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## 8. Contrato de respuesta

```json
{
  "prediction": "yes",
  "probabilities": { "no": 0.367195, "yes": 0.632805 },
  "confidence": 0.632805,
  "threshold": 0.485318
}
```

`confidence` es la mayor probabilidad de clase, no una garantía de corrección. El endpoint valida tipos, rangos básicos, campos faltantes y campos adicionales.

## 9. Nivel del sistema y riesgos

El sistema se clasifica como **nivel 1 - Predicción**. Devuelve una estimación para apoyar análisis humano, pero no recomienda una acción específica, no contacta clientes y no modifica sistemas externos.

Riesgos principales:

- falsos positivos que desperdicien capacidad de campaña;
- falsos negativos que omitan clientes potencialmente interesados;
- sesgo histórico o baja representatividad de segmentos;
- deriva de datos y cambios en campañas o canales;
- probabilidades no necesariamente calibradas;
- uso indebido de la predicción como decisión automática.

Controles recomendados: revisión humana, límites de uso, monitoreo de distribución y tasa de positivos, evaluación posterior con etiquetas reales, análisis por segmentos y reentrenamiento controlado. No se debe reincorporar `duration` ni variables disponibles solamente después del contacto.

## 10. Operación y costos

Cloud Run escala a cero con `min-instances=0`, pero Cloud Build, Artifact Registry y las invocaciones pueden generar consumo facturable según la cuenta. Para eliminar el servicio cuando ya no lo usemos (final del lab):

```powershell
gcloud run services delete bank-marketing-api --region us-central1
```
