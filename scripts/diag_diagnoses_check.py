import pandas as pd
import os
mimic_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MIMIC-3'))
path = os.path.join(mimic_path, 'DIAGNOSES_ICD.csv')
print('Loading', path)
df = pd.read_csv(path, nrows=20)
print('\nColumns:')
print(df.columns.tolist())
print('\nUppercased columns (notebook convention):')
print([c.upper() for c in df.columns.tolist()])
print('\nFirst rows:')
print(df.head().to_string(index=False))
# check presence
for candidate in ['HADM_ID','hadm_id','hadmid']:
    print(f"has '{candidate}'?", candidate in df.columns or candidate.upper() in [c.upper() for c in df.columns])
