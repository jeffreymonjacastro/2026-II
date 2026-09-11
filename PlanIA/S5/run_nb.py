import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import pandas as pd
import numpy as np

with open('05_Alumno_baseline_leakage_correlation-1.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Stop execution before placeholders
for i, cell in enumerate(nb.cells):
    if '6. Preprocesamiento' in cell.source:
        nb.cells = nb.cells[:i]
        break

# Fix cell 4 (Separar X/y)
for cell in nb.cells:
    if '4. Separar X/y' in cell.source:
        cell.source = """# ============================================================
# 4. Separar X/y
# ============================================================
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]"""
        
    if '5. Auditor' in cell.source:
        cell.source = """# ============================================================
# 5. Auditoría rápida de columnas
# ============================================================
summary = pd.DataFrame({
    "dtype": X_train.dtypes.astype(str),
    "n_missing": X_train.isnull().sum(),
    "pct_missing": (X_train.isnull().mean() * 100).round(2),
    "n_unique": X_train.nunique(),
}).sort_values(["dtype", "n_unique"])

display(summary.head(30))

numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

print("Numéricas:", len(numeric_cols))
print("Categóricas:", len(categorical_cols))"""

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
try:
    ep.preprocess(nb, {'metadata': {'path': './'}})
except Exception as e:
    print('Execution stopped:', e)

for cell in nb.cells:
    if 'EDA-12' in cell.source:
        for out in cell.outputs:
            if out.output_type == 'stream':
                print(out.text)
            elif hasattr(out, 'data') and 'text/plain' in out.data:
                print(out.data['text/plain'])
