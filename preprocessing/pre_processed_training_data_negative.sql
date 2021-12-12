WITH negative_records AS (
    SELECT n.subject_id,
           COUNT(DISTINCT n.chartdate) AS                 num_days,
           string_agg(d.icd9_code)                        icd_code_string,
           farm_fingerprint(CAST(n.subject_id AS string)) hashing
    FROM `physionet-data.mimiciii_notes.noteevents` n
             JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` d ON
        (n.subject_id = n.subject_id AND n.hadm_id = d.hadm_id)
    WHERE n.iserror IS NULL
    GROUP BY n.subject_id
    HAVING string_agg(d.icd9_code) NOT LIKE '%,428%'
       AND string_agg(d.icd9_code) NOT LIKE '428%'
)
   , no_prior_hf_patients AS (
    SELECT a.admittime,
           a.subject_id,
           a.hadm_id,
           string_agg(b.icd9_code)                                           AS icd9_code_list,
           CASE WHEN string_agg(b.icd9_code) LIKE '%,428%' THEN 1 ELSE 0 END AS hfdetectedflag,
           e.hashing
    FROM `physionet-data.mimiciii_clinical.admissions` a
             JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` b
                  ON a.subject_id = b.subject_id AND a.hadm_id = b.hadm_id
             JOIN `physionet-data.mimiciii_clinical.d_icd_diagnoses` c ON b.icd9_code = c.icd9_code
             JOIN negative_records e ON a.subject_id = e.subject_id -- get the negative  record patients
    GROUP BY a.admittime, a.subject_id, a.hadm_id, e.hashing
    HAVING hfdetectedflag = 0
    ORDER BY e.hashing
    limit 4000 -- limit number of records to somewhat the same as positive record we have
    )
   , raw_training_data AS (
SELECT P.*, N.category, N.text
FROM no_prior_HF_patients P
    JOIN `physionet-data.mimiciii_notes.noteevents` N
ON P.SUBJECT_ID=N.SUBJECT_ID AND P.HADM_ID=N.HADM_ID
    ), pre_processed_training_data AS (
SELECT *,
    RegexP_REPLACE( -- remove commas
    RegexP_REPLACE( -- remove redacted
    LOWER (text), r "\[\*\*(.*)\*\*\]", ""),
    r ",", "") AS clean_text
FROM raw_training_data
WHERE category = "Discharge summary"
    )

SELECT *
FROM pre_processed_training_data;