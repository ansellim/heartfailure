-- Daniel Tan
-- --- Counts of numbers of patients in the database
SELECT COUNT(DISTINCT a.subject_id), --46530
       COUNT(DISTINCT a.hadm_id)     -- 58976
FROM `physionet-data.mimiciii_clinical.admissions` a

SELECT COUNT(*),                     -- 651047
       COUNT(DISTINCT d.subject_id), --46520
       COUNT(DISTINCT d.hadm_id)     --58976
FROM `physionet-data.mimiciii_clinical.diagnoses_icd` d

--- A patient (subject_id) may have multiple admission records (HADM_ID)
SELECT a.subject_id, COUNT(DISTINCT a.hadm_id) AS distinct_admissions
FROM `physionet-data.mimiciii_clinical.admissions` a
GROUP BY 1
ORDER BY 2 DESC

--- A patient (subject_id) may have multiple diagnoses(ICD_codes), but these will be ranked by SEQ_NUM
SELECT d.subject_id, COUNT(DISTINCT d.icd9_code) AS distinct_icd_codes
FROM `physionet-data.mimiciii_clinical.diagnoses_icd` d
GROUP BY 1
ORDER BY 2 DESC

-- Counts of ICD=428 (heart failure)
SELECT countif(d.icd9_code LIKE '428%')                                AS all_428_diagnoses, --20676
       COUNT(DISTINCT if(d.icd9_code LIKE '428%', d.subject_id, NULL)) AS patients_w_428,    --10154
       COUNT(DISTINCT if(d.icd9_code LIKE '428%', d.hadm_id, NULL))    AS admissions_w_428,  --13608
       COUNT(DISTINCT d.subject_id)                                    AS total_patients,    --46520
       COUNT(DISTINCT d.hadm_id)                                       AS total_admissions   --58976
FROM `physionet-data.mimiciii_clinical.diagnoses_icd` d;
