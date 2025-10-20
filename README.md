# Proyecto: SVM para análisis de Airbnb (Madrid)

Breve recibo y plan
- Tarea: Generar un README profesional basado en el código y comentarios disponibles en los notebooks y archivos del proyecto.
- Plan: 1) Descripción del objetivo y dataset. 2) Estructura de archivos. 3) Resumen de los procesos principales (preprocesamiento, feature engineering, pipelines). 4) Modelos y evaluación (clasificación y regresión). 5) Instrucciones rápidas para reproducir y próximos pasos.

Checklist (lo que encontrarás en este README)
- [x] Objetivo del proyecto
- [x] Estructura de carpetas y archivos clave
- [x] Descripción de los notebooks y funciones principales
- [x] Flujo de procesamiento y decisiones de preprocesado
- [x] Resumen de modelos, métricas y resultados esperados
- [x] Requisitos y cómo ejecutar los notebooks
- [x] Siguientes pasos recomendados

1. Objetivo

Este repositorio contiene un análisis y una práctica educativa sobre el uso de Support Vector Machines (SVM) aplicadas a dos tareas sobre un dataset de anuncios de Airbnb en Madrid:

- Clasificación: predecir `room_type` (tipo de alojamiento) usando `LinearSVC` y `SVC` (kernel RBF). Afinación por GridSearch para SVC.
- Regresión: predecir `price` (precio por noche) usando `LinearSVR` y `SVR` (kernel RBF). Transformación logarítmica del objetivo y GridSearch para SVR.

El propósito es comparar modelos lineales vs no lineales, aplicar buen preprocesamiento y pipelines reproducibles.

2. Estructura del proyecto

- README.md  (este archivo)
- context/
  - Actividad_Airbnb-SVC.ipynb  — enunciado y resolución de la actividad (descripción general)
- data/
  - airbnb.csv  — dataset principal (ubicación esperada: `data/airbnb.csv`)
- notebooks/
  - svc.ipynb  — notebook de clasificación (LinearSVC vs SVC, pipelines y evaluación)
  - svr.ipynb  — notebook de regresión (LinearSVR vs SVR, transformación de `price`, GridSearch)
- .gitignore, .idea/, etc. — archivos de entorno

3. Columnas y variables relevantes (detectadas en los notebooks)

Variables usadas frecuentemente:
- price (objetivo en regresión)
- minimum_nights
- number_of_reviews
- reviews_per_month
- calculated_host_listings_count
- availability_365
- latitude, longitude (se usan para crear distance_to_center y luego se eliminan)
- neighbourhood (se sugiere eliminar por alta cardinalidad)
- neighbourhood_group
- room_type (objetivo en clasificación)

4. Flujo de procesamiento (resumen)

Basado en los notebooks, el flujo estándar sigue estos pasos:

- Carga del CSV: `data/airbnb.csv`.
- Muestreo (opcional) para acelerar experimentos: p. ej. `sample(n=13321)` o `sample(n=5000)` según notebook.
- Filtrado de outliers: uso de IQR (1.5) sobre columnas clave (`price`, `minimum_nights`, `calculated_host_listings_count`) y/o percentiles (5%/95%) para recortes más conservadores.
- Eliminación de columnas con alta cardinalidad o irrelevantes: `neighbourhood`, y eventualmente `latitude`/`longitude` después de crear features derivadas.
- Feature engineering:
  - `distance_to_center`: distancia a Puerta del Sol (coords fijas) calculada con haversine a partir de lat/lon.
  - `price_category`: segmentación del precio en bins (budget/moderate/premium/luxury).
- Imputación y escalado:
  - SimpleImputer (median) para numéricas.
  - StandardScaler para features numéricas (esencial para SVM).
  - OneHotEncoder para categóricas (`neighbourhood_group`, `price_category`) con `drop='first'` y `handle_unknown='ignore'`.
- Codificación del target de clasificación con `LabelEncoder`.

5. Pipelines y componentes clave (notebooks)

Funciones / clases que aparecen y su propósito:

- FeatureEngineer (custom transformer): crea `distance_to_center`, `price_category` y elimina lat/lon.
- load_and_preprocess_data(): carga, muestrea y aplica filtros de outliers; devuelve DataFrame preprocesado.
- build_pipeline(): construye Pipeline completo con `FeatureEngineer` + `ColumnTransformer` + modelo (SVC/SVR).
- evaluate_model_cv() / cv_metrics(): evaluación con cross-validation (KFold o StratifiedKFold) retornando métricas agregadas.
- Main / GridSearchCV: se usa GridSearchCV para buscar `C` y `gamma` (y `epsilon` para SVR) optimizando `f1_macro` (clasificación) o `neg_root_mean_squared_error` (regresión).

6. Modelos y métricas (resumen de resultados observados)

Clasificación (SVC vs LinearSVC)
- Métricas: Accuracy, F1-Score (macro), Precision (macro), Recall (macro)
- Observación: SVC (RBF) mostró mejora pequeña pero consistente (~+2% en accuracy y F1) respecto a LinearSVC, indicando cierta no linealidad en los datos.
- Nota: la clase `Shared room` es muy minoritaria y penaliza las métricas macro.

Regresión (SVR vs LinearSVR)
- Métricas: RMSE, MAE, R² (se aplica transformación log a `price` durante el entrenamiento y se vuelve a transformar para evaluación)
- Observación: diferencias pequeñas entre modelos; R² ~ 0.46–0.48 y RMSE ~ 22 (unidad: euros en escala original), lo que sugiere que con las features actuales aún queda varianza sin explicar.

7. Requisitos y cómo reproducir (instrucciones rápidas)

Requisitos sugeridos (crear `venv` recomendado):
- Python 3.8+
- Paquetes principales: pandas, numpy, scikit-learn, haversine, jupyterlab / notebook

Ejemplo de instalación (en Windows `cmd.exe`):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install pandas numpy scikit-learn haversine jupyterlab
```

Cómo ejecutar los notebooks:
- Abrir JupyterLab/Jupyter Notebook desde la raíz del proyecto (donde está `data/airbnb.csv`).
- Cargar y ejecutar `notebooks/svc.ipynb` para clasificación.
- Cargar y ejecutar `notebooks/svr.ipynb` para regresión.

Recomendaciones de ejecución
- Si el dataset es grande, ejecutar los notebooks inicialmente con la opción de muestreo (ej. `sample(n=1000)`) para iterar rápido.
- Ajustar `n_jobs=-1` en GridSearchCV sólo si la máquina dispone de CPU/memoria suficiente.

8. Contrato rápido (inputs/outputs/errores esperados)

- Input: `data/airbnb.csv` (CSV con columnas mencionadas). Notebooks esperan la ruta `../data/airbnb.csv` relativa al notebook.
- Output: resultados de evaluación (métricas CV), mejores hiperparámetros, y pipelines entrenados (si se guardan manualmente).
- Errores comunes: falta de archivo `airbnb.csv`, memoria insuficiente en GridSearchCV, dependencia `haversine` ausente.

9. Casos borde a considerar

- Clases muy desbalanceadas (Shared room) → usar métricas macro, `class_weight='balanced'` o técnicas de re-muestreo.
- Valores cero en `reviews_per_month` / `number_of_reviews` / `availability_365` → se reemplazan por NaN para imputación.
- Valores extremos en `price` y `minimum_nights` → aplicar IQR o percentiles antes de entrenar.
- High-cardinality: `neighbourhood` puede causar explosión dimensional si se one-hot-encoda; se propone eliminarla o agrupar.

---
