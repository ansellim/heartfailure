--- see the record on how many patient has admission that without HF diagnosis, but get diagnosed HF later..

WITH positive_records as (
    SELECT 
        distinct n.SUBJECT_ID,
        count(distinct n.CHARTDATE) as num_days
    FROM `physionet-data.mimiciii_notes.noteevents` n
    JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` d ON 
        (n.SUBJECT_ID=n.SUBJECT_ID AND n.HADM_ID=d.HADM_ID)
    WHERE 
        n.ISERROR is null
        AND d.ICD9_CODE like '428%'
    GROUP BY 1
    limit 10 -- limit 10 patients
)

select A.ADMITTIME,A.SUBJECT_ID,A.HADM_ID, String_agg(B.ICD9_CODE) as ICD9_CODE_LIST,
case when String_agg(B.ICD9_CODE) like '%,428%' then 1 else 0 end as HFDetectedFlag
from  `physionet-data.mimiciii_clinical.admissions`  A 
join `physionet-data.mimiciii_clinical.diagnoses_icd` B on A.SUBJECT_ID =B.SUBJECT_ID and A.HADM_ID=B.HADM_ID
join `physionet-data.mimiciii_clinical.d_icd_diagnoses` C on B.ICD9_CODE = C.ICD9_CODE
join positive_records E on A.SUBJECT_ID =E.SUBJECT_ID  -- get the positive 428 record patients
group by A.ADMITTIME,A.SUBJECT_ID,A.HADM_ID
order by 2,1;


---how many patients has no HF diagnosis in early admission but detected later
WITH final_result as (

    WITH intermediate_result as (
        
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
        )

        select A.ADMITTIME,A.SUBJECT_ID,A.HADM_ID, String_agg(B.ICD9_CODE) as ICD9_CODE_LIST,
        case when String_agg(B.ICD9_CODE) like '%,428%' then 1 else 0 end as HFDetectedFlag
        from  `physionet-data.mimiciii_clinical.admissions`  A 
        join `physionet-data.mimiciii_clinical.diagnoses_icd` B on A.SUBJECT_ID =B.SUBJECT_ID and A.HADM_ID=B.HADM_ID
        join `physionet-data.mimiciii_clinical.d_icd_diagnoses` C on B.ICD9_CODE = C.ICD9_CODE
        join positive_records E on A.SUBJECT_ID =E.SUBJECT_ID  -- get the positive 428 record patients
        group by A.ADMITTIME,A.SUBJECT_ID,A.HADM_ID
        )


    Select subA.*,subB.Note_record_count_wo_HF,subB.Note_record_count_with_HF
    From
    (Select a.SUBJECT_ID ,count(*) total_admit, sum(HFDetectedFlag) as HFDetectedAdmission
    from intermediate_result a
    group by a.SUBJECT_ID
    ) subA
    join 
    (
    SELECT
    a.SUBJECT_ID ,
    sum(case when HFDetectedFlag = 0 then 1 else 0 end) Note_record_count_wo_HF,
    sum(case when HFDetectedFlag = 1 then 1 else 0 end) Note_record_count_with_HF
    from intermediate_result a
    join `physionet-data.mimiciii_notes.noteevents` n on a.SUBJECT_ID =n.SUBJECT_ID and a.HADM_ID=n.HADM_ID 
    group by a.SUBJECT_ID
    ) subB on subA.SUBJECT_ID=subB.SUBJECT_ID

    )

select  '0_count of patients that have no HF in early admission' as category , count(*) as cnt 
, avg(total_admit-HFDetectedAdmission) MedianAdmitWithoutHFDiagnosos, avg(total_admit) MedianTotalAdmission
, avg(Note_record_count_wo_HF) avgNote_record_count_wo_HF
, avg(Note_record_count_with_HF) avgNote_record_count_with_HF
from final_result r
where total_admit>HFDetectedAdmission
union all 
select  '1_count of patients that have  HF since the start of admission' as category , count(*) as cnt 
, avg(total_admit-HFDetectedAdmission) AvgAdmitWithoutHFDiagnosos, avg(total_admit) AvgTotalAdmission
, avg(Note_record_count_wo_HF) avgNote_record_count_wo_HF
, avg(Note_record_count_with_HF) avgNote_record_count_with_HF
from final_result
where total_admit=HFDetectedAdmission
union all 
select '2_total', count(distinct subject_ID)
, avg(total_admit-HFDetectedAdmission) AvgAdmitWithoutHFDiagnosos, avg(total_admit) AvgTotalAdmission
, avg(Note_record_count_wo_HF) avgNote_record_count_wo_HF
, avg(Note_record_count_with_HF) avgNote_record_count_with_HF
from final_result
order by 1
;




