import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Create toy dataset to reproduce groupwise imputation
df = pd.DataFrame({
    'patient_id': ['p1']*3 + ['p2']*2,
    'HR': [80, np.nan, 78, np.nan, 85],
    'Temp': [36.6, 37.1, np.nan, 38.0, np.nan],
    'SepsisLabel': [0, 0, 1, 0, 1]
})

print('Original DF:')
print(df)

imputed_data = []
for name, group in df.groupby('patient_id'):
    imputed_group = group.copy()
    imputed_group = imputed_group.ffill().bfill()
    numeric_cols = imputed_group.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        imputer = SimpleImputer(strategy='median')
        imputed_vals = imputer.fit_transform(imputed_group[numeric_cols])
        imputed_group.loc[:, numeric_cols] = imputed_vals
    imputed_data.append(imputed_group)

imputed_combined = pd.concat(imputed_data, ignore_index=True)
print('\nImputed DF:')
print(imputed_combined)
print('\nRemaining missing values:', imputed_combined.isna().sum().sum())
