WITH negative_records as (
    SELECT 
        n.SUBJECT_ID,
        count(distinct n.CHARTDATE) as num_days,String_agg(d.ICD9_CODE) ICD_Code_String,
        FARM_FINGERPRINT(cast(n.SUBJECT_ID as string)) Hashing
    FROM `physionet-data.mimiciii_notes.noteevents` n
    JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` d ON 
        (n.SUBJECT_ID=n.SUBJECT_ID AND n.HADM_ID=d.HADM_ID)
    WHERE 
        n.ISERROR is null
    GROUP BY n.SUBJECT_ID 
    having String_agg(d.ICD9_CODE) not like '%,428%' and String_agg(d.ICD9_CODE) not like '428%' 
), no_prior_HF_patients as (
    select A.ADMITTIME,A.SUBJECT_ID,A.HADM_ID, String_agg(B.ICD9_CODE) as ICD9_CODE_LIST,
    case when String_agg(B.ICD9_CODE) like '%,428%' then 1 else 0 end as HFDetectedFlag, E.Hashing
    from  `physionet-data.mimiciii_clinical.admissions`  A 
    join `physionet-data.mimiciii_clinical.diagnoses_icd` B on A.SUBJECT_ID =B.SUBJECT_ID and A.HADM_ID=B.HADM_ID
    join `physionet-data.mimiciii_clinical.d_icd_diagnoses` C on B.ICD9_CODE = C.ICD9_CODE
    join negative_records E on A.SUBJECT_ID =E.SUBJECT_ID  -- get the negative  record patients
    group by A.ADMITTIME,A.SUBJECT_ID,A.HADM_ID, E.Hashing
    HAVING HFDetectedFlag = 0
    order by E.Hashing limit 4000 -- limit number of records to somewhat the same as positive record we have
), raw_training_data as (
    select P.*,N.category,N.text 
    from no_prior_HF_patients P
    join `physionet-data.mimiciii_notes.noteevents` N 
        on P.SUBJECT_ID=N.SUBJECT_ID and P.HADM_ID=N.HADM_ID
), pre_processed_training_data as (
    select *,
    RegexP_REPLACE( -- remove commas
        RegexP_REPLACE( -- remove redacted
            LOWER(text), r"\[\*\*(.*)\*\*\]",""),
        r",","")  as clean_text
    from raw_training_data 
    where category = "Discharge summary"
)

select *  from pre_processed_training_data;