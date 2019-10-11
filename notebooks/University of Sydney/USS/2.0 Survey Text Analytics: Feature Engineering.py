# Databricks notebook source
# MAGIC %matplotlib inline
# MAGIC %config InlineBackend.figure_format = 'retina'
# MAGIC 
# MAGIC import pandas as pd
# MAGIC import numpy as np
# MAGIC import matplotlib.pyplot as plt
# MAGIC 
# MAGIC np.random.seed(123456)

# COMMAND ----------

# MAGIC %scala
# MAGIC val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
# MAGIC val hiveDF = hiveContext.sql("select *,1 as num from survey.usstrainingView")
# MAGIC //val personnelTable = spark.catalog.getTable("survey.usstrainingView")
# MAGIC hiveDF.createOrReplaceTempView("usstrainingView")

# COMMAND ----------

articles_all = sqlContext.table("usstrainingView")
dataset = articles_all.toPandas()

# COMMAND ----------

dataset.head()

# COMMAND ----------

# DBTITLE 1,pivot table example with three columns
# pivot table example with three columns
datasetpivot=pd.pivot_table(dataset, 
                            values='num', 
                            index=['text'], 
                            columns='parent',
                            aggfunc='sum',fill_value=0
              ).reset_index()

# COMMAND ----------

datasetpivot

# COMMAND ----------

word_stats = spark.createDataFrame(pd.DataFrame(datasetpivot, columns=['text','AD','CU','GQ','LC','LE','LR','SU','TQ']))
word_stats.createOrReplaceTempView('datasetpivot')

# COMMAND ----------

# MAGIC %scala
# MAGIC var wordStatsSpark = sqlContext.table("datasetpivot")
# MAGIC wordStatsSpark.write.mode("overwrite").saveAsTable("survey.usstrainingPivot")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from survey.usstrainingPivot

# COMMAND ----------

dataset['parent'].unique()

# COMMAND ----------

dataset['parent'].value_counts().plot(kind="bar")
plt.show()
display()

# COMMAND ----------

df_toxic = datasetpivot.drop(['text'], axis=1)
counts = []
categories = list(df_toxic.columns.values)
for i in categories:
    counts.append((i, df_toxic[i].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])
df_stats

# COMMAND ----------

df_stats.plot(x='category', y='number_of_comments', kind='bar', legend=False, grid=True, figsize=(8, 5))
plt.title("Number of comments per category")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('category', fontsize=12)
display()

# COMMAND ----------

import seaborn as sns
rowsums = datasetpivot.iloc[:,1:].sum(axis=1)
x=rowsums.value_counts()

#plot
plt.figure(figsize=(8,5))
ax = sns.barplot(x.index, x.values)
plt.title("Multiple categories per comment")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('# of categories', fontsize=12)
display()

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from survey.usstrainingPivot where text='tutorials''

# COMMAND ----------

# MAGIC %sql
# MAGIC select text,parent,time,Document_ID from (
# MAGIC select  LOWER(text) as text,parent,count(*) as time,row_number() over(partition by LOWER(text) ORDER BY count(*) desc) AS Document_ID from survey.usstraining
# MAGIC group by LOWER(text),parent
# MAGIC having 
# MAGIC --time>1 and
# MAGIC Parent in ('TQ','CU','GQ','LE','LR','AD','LC','SU')
# MAGIC and LOWER(text)='placement'
# MAGIC order by time desc
# MAGIC )
# MAGIC where Document_ID=1

# COMMAND ----------

# MAGIC %sql
# MAGIC --select text,parent from(
# MAGIC select LOWER(text) as text,parent,count(*) as time from survey.usstraining
# MAGIC group by LOWER(text) ,parent
# MAGIC having time=1
# MAGIC and Parent in ('TQ','CU','GQ','LE','LR','AD','LC','SU')
# MAGIC and LOWER(text)='tutorials'
# MAGIC order by time desc
# MAGIC --)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from survey.usstraining

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from survey.usstraining1

# COMMAND ----------

# MAGIC %sql
# MAGIC --create or replace view trainingduplicatedview as
# MAGIC select a.*,b.parent as bparent,b.category as bcategory, b.text as btext,b.year as byear,b.area from survey.usstraining a
# MAGIC inner join survey.usstraining1 b
# MAGIC where a.text=b.text and a.category<>b.category

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from trainingduplicatedview

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE VIEW  survey.usstrainingallview as
# MAGIC select Parent,Category,text from survey.usstraining
# MAGIC union 
# MAGIC select Parent,Category,text from survey.usstraining1

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from survey.usstrainingView1level3

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE VIEW  survey.usstrainingView1 as
# MAGIC select text,parent,time,Document_ID from (
# MAGIC select  LOWER(text) as text,parent,count(*) as time,row_number() over(partition by LOWER(text) ORDER BY count(*) desc) AS Document_ID from survey.usstrainingallview
# MAGIC group by LOWER(text),parent
# MAGIC having 
# MAGIC --time>1 and
# MAGIC Parent in ('TQ','CU','GQ','LE','LR','AD','LC','SU')
# MAGIC 
# MAGIC order by time desc
# MAGIC )
# MAGIC where Document_ID=1

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE VIEW  survey.usstrainingView1level3 as
# MAGIC select distinct parent,Category,text from (
# MAGIC select a.parent,b.Category,a.text from survey.usstrainingView1 a
# MAGIC left join survey.usstrainingallview b on LOWER(a.parent)=LOWER(b.Parent) and LOWER(a.text)=LOWER(b.text)
# MAGIC ) c

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from survey.usstrainingView1level3 --order by 1,2 -- where category is null
# MAGIC --where Parent in ('TQ','CU','GQ','LE','LR','AD','LC','SU')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from survey.usstrainingview 

# COMMAND ----------

