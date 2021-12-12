-- George Seah

WITH positive_records AS (
    SELECT n.SUBJECT_ID,
           COUNT(DISTINCT n.CHARTDATE) AS num_days
    FROM `physionet-data.mimiciii_notes.noteevents` n
             JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` d ON
        (n.SUBJECT_ID = n.SUBJECT_ID AND n.HADM_ID = d.HADM_ID)
    WHERE n.ISERROR IS NULL
      AND d.ICD9_CODE LIKE '428%'
    GROUP BY 1
),
     no_prior_HF_patients AS (
         SELECT A.ADMITTIME,
                A.SUBJECT_ID,
                A.HADM_ID,
                STRING_AGG(B.ICD9_CODE)                                           AS ICD9_CODE_LIST,
                CASE WHEN STRING_AGG(B.ICD9_CODE) LIKE '%,428%' THEN 1 ELSE 0 END AS HFDetectedFlag
         FROM `physionet-data.mimiciii_clinical.admissions` A
                  JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` B
                       ON A.SUBJECT_ID = B.SUBJECT_ID AND A.HADM_ID = B.HADM_ID
                  JOIN `physionet-data.mimiciii_clinical.d_icd_diagnoses` C ON B.ICD9_CODE = C.ICD9_CODE
                  JOIN positive_records E ON A.SUBJECT_ID = E.SUBJECT_ID -- get the positive 428 record patients
         GROUP BY A.ADMITTIME, A.SUBJECT_ID, A.HADM_ID
         HAVING HFDetectedFlag = 0
     ),
     raw_training_data AS (
         SELECT *
         FROM no_prior_HF_patients P
                  JOIN `physionet-data.mimiciii_notes.noteevents` N
                       ON P.SUBJECT_ID = N.SUBJECT_ID AND P.HADM_ID = N.HADM_ID
     ),
     pre_processed_training_data AS (
         SELECT P.*,
                N.category,
                N.text, -- geo update to avoid duplicate subject_id and hadm_id
                REGEXP_REPLACE( -- remove commas
                        REGEXP_REPLACE( -- remove redacted
                                LOWER(text), r"\[\*\*(.*)\*\*\]", ""),
                        r",", "") AS clean_text
         FROM raw_training_data
         WHERE category = "Discharge summary"
     )

SELECT *
FROM pre_processed_training_data
