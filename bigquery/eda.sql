--- Counts
select 
count(distinct a.subject_id), --46530
count(distinct a.HADM_ID) -- 58976
FROM `physionet-data.mimiciii_clinical.admissions` a

select count(*), -- 651047
count(distinct d.subject_id), --46520
count(distinct d.HADM_ID) --58976
FROM `physionet-data.mimiciii_clinical.diagnoses_icd` d

--- A patient (subject_id) may have multiple admission records (HADM_ID)
select a.SUBJECT_ID, count(distinct a.HADM_ID) as distinct_admissions
FROM `physionet-data.mimiciii_clinical.admissions` a
GROUP BY 1
ORDER BY 2 DESC

--- A patient (subject_id) may have multiple diagnoses(ICD_codes), but these will be ranked by SEQ_NUM
select d.SUBJECT_ID, count(distinct d.ICD9_CODE) as distinct_ICD_codes
FROM `physionet-data.mimiciii_clinical.diagnoses_icd` d
GROUP BY 1
ORDER BY 2 DESC

-- Counts of ICD=428
SELECT 
    COUNTIF(d.ICD9_CODE like '428%') as all_428_diagnoses, --20676
    COUNT(DISTINCT IF(d.ICD9_CODE like '428%', d.SUBJECT_ID, NULL)) as patients_w_428, --10154
    COUNT(DISTINCT IF(d.ICD9_CODE like '428%', d.HADM_ID, NULL)) as admissions_w_428, --13608
    COUNT(distinct d.subject_id) as total_patients, --46520
    COUNT(distinct d.HADM_ID) as total_admissions --58976
FROM `physionet-data.mimiciii_clinical.diagnoses_icd` d;
