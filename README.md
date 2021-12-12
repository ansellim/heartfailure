# Heart Failure Prediction

Long Title: Applying Natural Language Processing Methods To Predict The Development Of Heart Failure In Patients In The Mimic-Iii Database: Comparison Of Classical Machine Learning Approaches With Deep Learning Methods

Authors: Ansel Lim, Daniel Tan, George Seah, Joanne Lee

Emails: ylim70@gatech.edu, ddanieltan@gatech.edu, jseah3@gatech.edu, jlee3702@gatech.edu 

## Introduction

In this project, we retrospectively analyzed the discharge summary textual data in a subset of the Medical Information Mart for Intensive Care III (MIMIC-III) database which corresponded to a cohort of over 5,000 patients with and without heart failure. The patients which constituted this subset were identified using database queries run on the BigQuery platform. 

The textual data in these patients’ discharge summaries was preprocessed with standard natural language techniques such as lemmatization and stopword removal. Thereafter, the textual data was embedded in appropriate vector representations before predictive models were trained. 

Classical machine learning techniques (XGBoost and support vector machine) as well as deep learning models (a vanilla recurrent neural network and a transformer-based architecture, namely BERT) were applied using their implementations in popular machine learning and deep learning libraries. 

## Get started

### Clone this repository

Clone this repository into your local machine, a virtual machine instance, or Google Colab. Unless you enjoy watching paint dry, it is suggested that you run the deep learning aspects of the code on a machine with GPU.

### Set up conda environment

Run the following commands in the root of the project folder to create a conda environment that has the packages necessary to run our code.

`conda env create -f environment.yml`

Be sure to activate this environment by typing the following code:

`conda activate heartfailure`

### Download the dataset

To download the data from MIMIC-III, you need to be a registered user on PhysioNet, a repository of freely-available medical research data. 

To set up your data, run the following command and enter your password when prompted.

`wget -r -N -c -np --user <username> --ask-password https://physionet.org/files/mimiciii/1.4/`

The dataset will be downloaded into your copy of this repository.

```
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

## Run our code

The codebase is split into two parts: the machine learning approach and the deep learning approach. The machine learning (ML) approach is in the `ml` folder whereas the deep learning (DL) approach is in the `neural` folder.

### Run the ML workflow
* Main script can be found in ml/main.py to run entire workflow
* Data dependencies should be available in 
    1. ml/datasets
    2. pre-processing/dataset_lemma_avg_v3

### Run the DL workflow

The code for running the deep learning 