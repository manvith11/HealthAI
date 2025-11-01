import pandas as pd
import numpy as np
import os

mimic_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MIMIC-3'))
# Load small sample
patients = pd.read_csv(os.path.join(mimic_path, 'PATIENTS.csv'), nrows=200)
admissions = pd.read_csv(os.path.join(mimic_path, 'ADMISSIONS.csv'), nrows=200)

# Normalize headers to uppercase
patients.columns = patients.columns.str.upper()
admissions.columns = admissions.columns.str.upper()

print('patients columns sample:', patients.columns.tolist())
print('admissions columns sample:', admissions.columns.tolist())

# Merge using case-insensitive SUBJECT_ID
p_sub = next((c for c in patients.columns if c.upper()=='SUBJECT_ID'), None)
a_sub = next((c for c in admissions.columns if c.upper()=='SUBJECT_ID'), None)
if p_sub and a_sub:
    pa = pd.merge(patients, admissions, left_on=p_sub, right_on=a_sub, how='inner', suffixes=('_PAT','_ADM'))
else:
    pa = pd.merge(patients, admissions, on='SUBJECT_ID', how='inner')

print('\npatient_admissions columns:')
print(pa.columns.tolist())

# Check for gender column variants
candidates = ['GENDER','GENDER_PAT','GENDER_ADM','SEX','SEX_PAT','SEX_ADM']
for cand in candidates:
    print(cand, 'in columns?', cand in pa.columns)

# If GENDER exists, print value counts; otherwise show head for diagnosis
if 'GENDER' in pa.columns:
    print('\nGENDER value_counts:')
    print(pa['GENDER'].value_counts(dropna=False))
else:
    print('\nNo literal GENDER column found; printing first few rows of merged frame:')
    print(pa.head().to_string(index=False))
