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
), no_prior_HF_patients as (
    select A.ADMITTIME,A.SUBJECT_ID,A.HADM_ID, String_agg(B.ICD9_CODE) as ICD9_CODE_LIST,
    case when String_agg(B.ICD9_CODE) like '%,428%' then 1 else 0 end as HFDetectedFlag
    from  `physionet-data.mimiciii_clinical.admissions`  A 
    join `physionet-data.mimiciii_clinical.diagnoses_icd` B on A.SUBJECT_ID =B.SUBJECT_ID and A.HADM_ID=B.HADM_ID
    join `physionet-data.mimiciii_clinical.d_icd_diagnoses` C on B.ICD9_CODE = C.ICD9_CODE
    join positive_records E on A.SUBJECT_ID =E.SUBJECT_ID 
    group by A.ADMITTIME,A.SUBJECT_ID,A.HADM_ID
    HAVING HFDetectedFlag = 0
), raw_training_data as (
    select * 
    from no_prior_HF_patients P
    join `physionet-data.mimiciii_notes.noteevents` N 
        on P.SUBJECT_ID=N.SUBJECT_ID and P.HADM_ID=N.HADM_ID
), pre_processed_training_data as (
    select *,
    RegexP_REPLACE(LOWER(text), r"\[\*\*(.*)\*\*\]","")  as clean_text
    from raw_training_data 
    where category = "Discharge summary"
)

select * from pre_processed_training_data

