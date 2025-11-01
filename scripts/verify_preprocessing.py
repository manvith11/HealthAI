import pandas as pd
import numpy as np
import os

mimic_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MIMIC-3'))
# Load small samples to verify flags and age logic (use same nrows used in notebook)
patients = pd.read_csv(os.path.join(mimic_path, 'PATIENTS.csv'), nrows=1000)
admissions = pd.read_csv(os.path.join(mimic_path, 'ADMISSIONS.csv'), nrows=500)
# Normalize columns
patients.columns = patients.columns.str.upper()
admissions.columns = admissions.columns.str.upper()
# Convert dates
for df, cols in [(patients, ['DOB','DOD']), (admissions, ['ADMITTIME','DISCHTIME','DEATHTIME'])]:
    for col in cols:
        match = next((c for c in df.columns if c.upper()==col.upper()), None)
        if match:
            df[match] = pd.to_datetime(df[match], errors='coerce')
# Add HAS_* flags
patients['HAS_DOB'] = patients[[c for c in patients.columns if c.upper()=='DOB']].notna().any(axis=1)
patients['HAS_DOD'] = patients[[c for c in patients.columns if c.upper()=='DOD']].notna().any(axis=1)
admissions['HAS_ADMITTIME'] = admissions[[c for c in admissions.columns if c.upper()=='ADMITTIME']].notna().any(axis=1)
admissions['HAS_DISCHTIME'] = admissions[[c for c in admissions.columns if c.upper()=='DISCHTIME']].notna().any(axis=1)
admissions['HAS_DEATHTIME'] = admissions[[c for c in admissions.columns if c.upper()=='DEATHTIME']].notna().any(axis=1)
# Merge and compute AGE using robust function
patient_admissions = pd.merge(patients, admissions, on='SUBJECT_ID', how='inner')

def compute_age(admit_ts, dob_ts):
    if pd.isna(admit_ts) or pd.isna(dob_ts):
        return np.nan
    age = admit_ts.year - dob_ts.year
    if (admit_ts.month, admit_ts.day) < (dob_ts.month, dob_ts.day):
        age -= 1
    return age

patient_admissions['AGE'] = patient_admissions.apply(lambda r: compute_age(r.get('ADMITTIME'), r.get('DOB')), axis=1)
print('HAS flags counts:')
print('patients HAS_DOB:', int(patients['HAS_DOB'].sum()), '/', len(patients))
print('admissions HAS_ADMITTIME:', int(admissions['HAS_ADMITTIME'].sum()), '/', len(admissions))
print('\nAGE summary:')
print(patient_admissions['AGE'].describe())
