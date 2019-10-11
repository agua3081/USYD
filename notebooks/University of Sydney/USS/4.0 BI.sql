-- Databricks notebook source
select distinct category from survey.usstraining 
where category not in 
(select category from survey.category)

-- COMMAND ----------

create view survey.usstrainingviewall as
select Parent,Category,text from survey.usstraining 

-- COMMAND ----------

select * from survey.usstrainingviewall

-- COMMAND ----------


--create view survey.usstrainingView as 
select * from 
(
select a.*,b.* from survey.usstraining a left join survey.category b on a.category=b.category
where a.Category <>' ' 
order by b.categoryid
) w

-- COMMAND ----------

--create view survey.usstrainingView as
select b.categoryid,a.category,a.text 
from survey.usstraining a 
left join survey.category b on a.category=b.category
order by 1


-- COMMAND ----------

select * from survey.category

-- COMMAND ----------

select count(*) from
(
select ID,o1 from survey.uss_data
WHERE cast(ID as double) is not null and o1 is not null
order by cast(ID as double)
) a
--where o1 is not null
--order by 1

-- COMMAND ----------

--create or replace view ussdata2017 as
select ID,o1 as text,substring(time_stamp,1,10) as date from survey.uss_data_1 where substring(time_stamp,1,4) in ('2016') and o1 is not null
union
select ID,o2 as text,substring(time_stamp,1,10) as date from survey.uss_data_1 where substring(time_stamp,1,4) in ('2016') and o2 is not null
order by 1

-- COMMAND ----------

select count(*) from ussdata2017

-- COMMAND ----------

select ID,o1,o2 from survey.uss_data WHERE cast(ID as double) is not null and o1 is not null order by cast(ID as double)

-- COMMAND ----------

create or replace view survey.USSdata as
select id,o1,'o1' as Area from survey.uss_data_1 
union 
select id,o2,'o2' as Area from survey.uss_data_1


-- COMMAND ----------

select * from survey.ussdata where o1 not in ('null','N/A.','N/A','...','Nil')
order by 1


-- COMMAND ----------

select text,parent,time,Document_ID from (
select  LOWER(text) as text,parent,count(*) as time,row_number() over(partition by LOWER(text) ORDER BY count(*) desc) AS Document_ID from survey.usstraining
group by LOWER(text),parent
having 
--time>1 and
Parent in ('TQ','CU','GQ','LE','LR','AD','LC','SU')

order by time desc
)
where Document_ID=1

-- COMMAND ----------

select * from survey.usstraining
where text not in(select text from survey.usstrainingView)

-- COMMAND ----------

select * from survey.usstrainingView

-- COMMAND ----------

select ID,o1 as text,'o1' as area  from survey.uss_data_1 where substring(time_stamp,1,4) in (2018,2019) and o1 not in('null','N/A.','N/A','...','Nil','None.','None','none')

-- COMMAND ----------

select * from survey.enr

-- COMMAND ----------

select * from survey.surveys

-- COMMAND ----------

select * from survey.uss_data_1

-- COMMAND ----------

select * from survey.enr where SID=420064703

-- COMMAND ----------

create or replace view powerbi.SurveyResponse as
select a.sid,a.o1,a.o2,a.complete,b.ID as SurveyID,b.dept as Department,b.dept_code,b.faculty,b.fullname as SurveyName,b.school,b.uos_code,b.uos_name from survey.uss_data_1 a
left join survey.surveys b on a.survey_key= b.survey_key


-- COMMAND ----------

select * from powerbi.SurveyResponse-- where survey_key='uss104730'

-- COMMAND ----------

select * from survey.uss_data_1 where survey_key='uss104730'

-- COMMAND ----------

select * from enrolments.students

-- COMMAND ----------

create or replace view powerbi.AnonymouseRespondent as
select sid,familyName,givenNames,countryOfOrigin,countryOfOriginDescription,languageSpokenAtHome,languageSpokenAtHomeDescription,internationalType,	internationalTypeDescription,isAusAid,distanceStudent,academicYear,source,dateOfBirth,studentType,studentTypeDescription
from enrolments.students

-- COMMAND ----------

select * from survey.latest_training

-- COMMAND ----------

