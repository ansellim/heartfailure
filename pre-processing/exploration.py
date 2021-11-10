# Ansel Lim
# 31 October 2021

import pandas as pd
import numpy as np
import pyspark.pandas as ps
from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("exploration").getOrCreate()
df = spark.read.csv("/tmp/resources/zipcodes.csv")


# Admissions
admissions = ps.read_csv('../physionet.org/files/mimiciii/1.4/admissions.csv')

# NOTEEVENTS: Deidentified notes, including nursing and physician notes, ECG reports, imaging reports, and discharge summaries.
# https://mimic.mit.edu/docs/iii/tables/
# https://mimic.mit.edu/docs/iii/tables/noteevents/
noteevents = ps.read_csv("../physionet.org/files/mimiciii/1.4/noteevents.csv")

# PATIENTS: EEvery unique patient in the database (defines SUBJECT_ID)
patients = ps.read_csv("../physionet.org/files/mimiciii/1.4/patients.csv")

# D_ICD_DIAGNOSES: Dictionary of International Statistical Classification of Diseases and Related Health Problems (ICD) codes relating to diagnoses
icd_diagnoses = ps.read_csv("../physionet.org/files/mimiciii/1.4/d_icd_diagnoses.csv")

# DIAGNOSES_ICD: Hospital assigned diagnoses, coded using the International Statistical Classification of Diseases and Related Health Problems (ICD) system
diagnoses_icd = ps.read_csv("../physionet.org/files/mimiciii/1.4/diagnoses_icd.csv")

# DRGCODES: Diagnosis Related Groups (DRG), which are used by the hospital for billing purposes.
drgcodes = ps.read_csv("../physionet.org/mimiciii/1.4/drgcodes.csv")

#####################

# Only use discharge summaries

