# CSE6250_Project

## Data setup
Daniel : I've set up a directory `physionet.org` that contains all the MIMIC3 data files like so.
abc
```
.
├── README.md
├── Useful_resources.txt
└── physionet.org
    ├── files
    │   └── mimiciii
    │       └── 1.4
    │           ├── ADMISSIONS.csv.gz
    │           ├── CALLOUT.csv.gz
    │           ├── CAREGIVERS.csv.gz
    │           ├── CHARTEVENTS.csv.gz
    │           ├── CPTEVENTS.csv.gz
    │           ├── DATETIMEEVENTS.csv.gz
    │           ├── DIAGNOSES_ICD.csv.gz
    │           ├── DRGCODES.csv.gz
    │           ├── D_CPT.csv.gz
    │           ├── D_ICD_DIAGNOSES.csv.gz
    │           ├── D_ICD_PROCEDURES.csv.gz
    │           ├── D_ITEMS.csv.gz
    │           ├── D_LABITEMS.csv.gz
    │           ├── ICUSTAYS.csv.gz
    │           ├── INPUTEVENTS_CV.csv.gz
    │           ├── INPUTEVENTS_MV.csv.gz
    │           ├── LABEVENTS.csv.gz
    │           ├── LICENSE.txt
    │           ├── MICROBIOLOGYEVENTS.csv.gz
    │           ├── NOTEEVENTS.csv.gz
    │           ├── OUTPUTEVENTS.csv.gz
    │           ├── PATIENTS.csv.gz
    │           ├── PRESCRIPTIONS.csv.gz
    │           ├── PROCEDUREEVENTS_MV.csv.gz
    │           ├── PROCEDURES_ICD.csv.gz
    │           ├── README.md
    │           ├── SERVICES.csv.gz
    │           ├── SHA256SUMS.txt
    │           ├── TRANSFERS.csv.gz
    │           └── index.html
    └── robots.txt
```
To set up the same locally run this command and fill in your physionet password when prompted.
`wget -r -N -c -np --user <username> --ask-password https://physionet.org/files/mimiciii/1.4/`

hello there from ansel

I've also added this directory to the `.gitignore` so the data files won't be git-tracked.
