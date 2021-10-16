-- what are the average/median/range of number of days of records we have for the two classes?
WITH positive_records as (
    SELECT 
        n.SUBJECT_ID,
        count(distinct n.CHARTDATE) as num_days
    FROM `physionet-data.mimiciii_notes.noteevents` n
    JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` d ON 
        (n.SUBJECT_ID=n.SUBJECT_ID AND n.HADM_ID=d.HADM_ID)
    WHERE 
        n.ISERROR is null
        AND d.ICD9_CODE like '428%'
    GROUP BY 1
), negative_records as (
    SELECT 
        n.SUBJECT_ID,
        count(distinct n.CHARTDATE) as num_days
    FROM `physionet-data.mimiciii_notes.noteevents` n
    JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` d ON 
        (n.SUBJECT_ID=n.SUBJECT_ID AND n.HADM_ID=d.HADM_ID)
    WHERE 
        n.ISERROR is null
        AND d.ICD9_CODE not like '428%'
    GROUP BY 1
)

SELECT
    'Positive' as label,
    COUNT(*) as patients,
    AVG(num_days) as avg_num_days,
    MIN(num_days) as min_num_days,
    MAX(num_days) as max_num_days
FROM positive_records

UNION ALL

SELECT
    'Negative' as label,
    COUNT(*) as patients,
    AVG(num_days) as avg_num_days,
    MIN(num_days) as min_num_days,
    MAX(num_days) as max_num_days
FROM negative_records

-- Percentile Distribution of records
WITH positive_records as (
    SELECT 
        n.SUBJECT_ID,
        count(distinct n.CHARTDATE) as num_days
    FROM `physionet-data.mimiciii_notes.noteevents` n
    JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` d ON 
        (n.SUBJECT_ID=n.SUBJECT_ID AND n.HADM_ID=d.HADM_ID)
    WHERE 
        n.ISERROR is null
        AND d.ICD9_CODE like '428%'
    GROUP BY 1
), negative_records as (
    SELECT 
        n.SUBJECT_ID,
        count(distinct n.CHARTDATE) as num_days
    FROM `physionet-data.mimiciii_notes.noteevents` n
    JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` d ON 
        (n.SUBJECT_ID=n.SUBJECT_ID AND n.HADM_ID=d.HADM_ID)
    WHERE 
        n.ISERROR is null
        AND d.ICD9_CODE not like '428%'
    GROUP BY 1
)
select
  'Positve' as label,
  percentiles[offset(10)] as p10,
  percentiles[offset(25)] as p25,
  percentiles[offset(50)] as p50,
  percentiles[offset(75)] as p75,
  percentiles[offset(90)] as p90,
  percentiles[offset(100)] as p100
from (
  select approx_quantiles(num_days, 100) as percentiles
  from positive_records
)

UNION ALL 

select
  'Negative' as label,
  percentiles[offset(10)] as p10,
  percentiles[offset(25)] as p25,
  percentiles[offset(50)] as p50,
  percentiles[offset(75)] as p75,
  percentiles[offset(90)] as p90,
  percentiles[offset(100)] as p100
from (
  select approx_quantiles(num_days, 100) as percentiles
  from negative_records
)
;
