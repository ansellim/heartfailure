-- Daniel Tan
-- Get summary statistics (mean, median, range) of number of days of records we have for the two classes (heart failure and no heart failure)
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
     negative_records AS (
         SELECT n.SUBJECT_ID,
                COUNT(DISTINCT n.CHARTDATE) AS num_days
         FROM `physionet-data.mimiciii_notes.noteevents` n
                  JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` d ON
             (n.SUBJECT_ID = n.SUBJECT_ID AND n.HADM_ID = d.HADM_ID)
         WHERE n.ISERROR IS NULL
           AND d.ICD9_CODE NOT LIKE '428%'
         GROUP BY 1
     )

SELECT 'Positive'    AS label,
       COUNT(*)      AS patients,
       AVG(num_days) AS avg_num_days,
       MIN(num_days) AS min_num_days,
       MAX(num_days) AS max_num_days
FROM positive_records

UNION ALL

SELECT 'Negative'    AS label,
       COUNT(*)      AS patients,
       AVG(num_days) AS avg_num_days,
       MIN(num_days) AS min_num_days,
       MAX(num_days) AS max_num_days
FROM negative_records

-- Percentile Distribution of records
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
     negative_records AS (
         SELECT n.SUBJECT_ID,
                COUNT(DISTINCT n.CHARTDATE) AS num_days
         FROM `physionet-data.mimiciii_notes.noteevents` n
                  JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` d ON
             (n.SUBJECT_ID = n.SUBJECT_ID AND n.HADM_ID = d.HADM_ID)
         WHERE n.ISERROR IS NULL
           AND d.ICD9_CODE NOT LIKE '428%'
         GROUP BY 1
     )
SELECT 'Positve'                AS label,
       percentiles[OFFSET(10)]  AS p10,
       percentiles[OFFSET(25)]  AS p25,
       percentiles[OFFSET(50)]  AS p50,
       percentiles[OFFSET(75)]  AS p75,
       percentiles[OFFSET(90)]  AS p90,
       percentiles[OFFSET(100)] AS p100
FROM (
         SELECT APPROX_QUANTILES(num_days, 100) AS percentiles
         FROM positive_records
     )

UNION ALL

SELECT 'Negative'               AS label,
       percentiles[OFFSET(10)]  AS p10,
       percentiles[OFFSET(25)]  AS p25,
       percentiles[OFFSET(50)]  AS p50,
       percentiles[OFFSET(75)]  AS p75,
       percentiles[OFFSET(90)]  AS p90,
       percentiles[OFFSET(100)] AS p100
FROM (
         SELECT APPROX_QUANTILES(num_days, 100) AS percentiles
         FROM negative_records
     )
;
