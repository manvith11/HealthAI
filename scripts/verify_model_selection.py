import pandas as pd
import os

mimic_path = os.path.join('..','MIMIC-3')

# load small samples
patients = pd.read_csv(os.path.join(mimic_path, 'PATIENTS.csv'), nrows=200)
admissions = pd.read_csv(os.path.join(mimic_path, 'ADMISSIONS.csv'), nrows=200)

# normalize headers
patients.columns = patients.columns.str.upper()
admissions.columns = admissions.columns.str.upper()

# minimal merge using SUBJECT_ID

def find_col(df, target):
    target_up = target.strip().upper()
    for c in df.columns:
        if c is None:
            continue
        if c.strip().upper() == target_up:
            return c
    return None

p_sub = find_col(patients, 'SUBJECT_ID')
a_sub = find_col(admissions, 'SUBJECT_ID')
if p_sub and a_sub:
    pa = pd.merge(patients, admissions, left_on=p_sub, right_on=a_sub, how='inner', suffixes=('_P','_A'))
else:
    print('Could not create patient_admissions; SUBJECT_ID not found')
    pa = pd.DataFrame()

# Desired features
features = [
    'SUBJECT_ID', 'HADM_ID', 'AGE', 'GENDER', 'ADMISSION_TYPE',
    'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE',
    'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY',
    'LOS_DAYS', 'NUM_DIAGNOSES', 'NUM_PROCEDURES',
    'HOSPITAL_MORTALITY', 'READMISSION_30D'
]

present = []
missing = []
for f in features:
    c = find_col(pa, f)
    if c:
        present.append(c)
    else:
        missing.append(f)

print('Present features:', present)
print('Missing features:', missing)

# attempt to run the same encoding logic from the notebook (simplified)
cols_to_encode = [c for c in ['GENDER','ADMISSION_TYPE','INSURANCE'] if find_col(pa,c)]
if cols_to_encode:
    print('Would encode:', cols_to_encode)
    try:
        encoded = pd.get_dummies(pa, columns=cols_to_encode, drop_first=True)
        print('Encoded shape:', encoded.shape)
    except Exception as e:
        print('Error during encoding:', e)
else:
    print('No categorical columns present to encode on sample')
