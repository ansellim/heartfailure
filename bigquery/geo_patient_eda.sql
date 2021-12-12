-- George Seah
-- Check how many patients do not have heart failure in one or more admissions, but eventually get admitted again with heart failure later on.
WITH positive_records AS (
    SELECT DISTINCT n.SUBJECT_ID,
                    COUNT(DISTINCT n.CHARTDATE) AS num_days
    FROM `physionet-data.mimiciii_notes.noteevents` n
             JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` d ON
        (n.SUBJECT_ID = n.SUBJECT_ID AND n.HADM_ID = d.HADM_ID)
    WHERE n.ISERROR IS NULL
      AND d.ICD9_CODE LIKE '428%'
    GROUP BY 1
    LIMIT 10 -- limit 10 patients
)

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
ORDER BY 2, 1;


WITH final_result AS (
    WITH intermediate_result AS (
        WITH positive_records AS (
            SELECT n.SUBJECT_ID,
                   COUNT(DISTINCT n.CHARTDATE) AS num_days
            FROM `physionet-data.mimiciii_notes.noteevents` n
                     JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` d ON
                (n.SUBJECT_ID = n.SUBJECT_ID AND n.HADM_ID = d.HADM_ID)
            WHERE n.ISERROR IS NULL
              AND d.ICD9_CODE LIKE '428%'
            GROUP BY 1
        )

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
    )

    SELECT subA.*, subB.Note_record_count_wo_HF, subB.Note_record_count_with_HF
    FROM (SELECT a.SUBJECT_ID, COUNT(*) total_admit, SUM(HFDetectedFlag) AS HFDetectedAdmission
          FROM intermediate_result a
          GROUP BY a.SUBJECT_ID
         ) subA
             JOIN
         (
             SELECT a.SUBJECT_ID,
                    SUM(CASE WHEN HFDetectedFlag = 0 THEN 1 ELSE 0 END) Note_record_count_wo_HF,
                    SUM(CASE WHEN HFDetectedFlag = 1 THEN 1 ELSE 0 END) Note_record_count_with_HF
             FROM intermediate_result a
                      JOIN `physionet-data.mimiciii_notes.noteevents` n
                           ON a.SUBJECT_ID = n.SUBJECT_ID AND a.HADM_ID = n.HADM_ID
             GROUP BY a.SUBJECT_ID
         ) subB ON subA.SUBJECT_ID = subB.SUBJECT_ID
)

SELECT '0_count of patients that have no HF in early admission' AS category
     , COUNT(*)                                                 AS cnt
     , AVG(total_admit - HFDetectedAdmission)                      MedianAdmitWithoutHFDiagnosos
     , AVG(total_admit)                                            MedianTotalAdmission
     , AVG(Note_record_count_wo_HF)                                avgNote_record_count_wo_HF
     , AVG(Note_record_count_with_HF)                              avgNote_record_count_with_HF
FROM final_result r
WHERE total_admit > HFDetectedAdmission
UNION ALL
SELECT '1_count of patients that have  HF since the start of admission' AS category
     , COUNT(*)                                                         AS cnt
     , AVG(total_admit - HFDetectedAdmission)                              AvgAdmitWithoutHFDiagnosos
     , AVG(total_admit)                                                    AvgTotalAdmission
     , AVG(Note_record_count_wo_HF)                                        avgNote_record_count_wo_HF
     , AVG(Note_record_count_with_HF)                                      avgNote_record_count_with_HF
FROM final_result
WHERE total_admit = HFDetectedAdmission
UNION ALL
SELECT '2_total'
     , COUNT(DISTINCT subject_ID)
     , AVG(total_admit - HFDetectedAdmission) AvgAdmitWithoutHFDiagnosos
     , AVG(total_admit)                       AvgTotalAdmission
     , AVG(Note_record_count_wo_HF)           avgNote_record_count_wo_HF
     , AVG(Note_record_count_with_HF)         avgNote_record_count_with_HF
FROM final_result
ORDER BY 1
;




