// Databricks notebook source
// MAGIC %md
// MAGIC #Data Ingestion Pipeline

// COMMAND ----------

// MAGIC %python
// MAGIC dbutils.widgets.dropdown("1", "1", [str(x) for x in range(1, 10)], "hello this is a widget")

// COMMAND ----------

spark.conf.set(
  "uosblob.azure.account.key.ipdipoc.blob.core.windows.net",
  "gyk86tWmHUT8VQRIE82Ms9RSsWYDgRinUiE0SSQY2iKBNOVGUo1h6/gVholNDvMrmTFQ3C296e9djou2MhtQIw==")

// COMMAND ----------

// MAGIC %fs ls wasbs://uss@uosblob.blob.core.windows.net/

// COMMAND ----------


import org.apache.spark.sql.SQLContext

// COMMAND ----------

val train = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("wholeFile", true)
    .option("multiline",true)
    .option("header", "true") // Use first line of all files as header
    .option("inferSchema", "true") // Automatically infer data types
    .option("escape", "\"")
    //.option("escape", "@")
    //.option("escape", "")
    .load("wasbs://uss@uosblob.blob.core.windows.net/train.csv")



// COMMAND ----------

display(train)

// COMMAND ----------

val Category = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("wholeFile", true)
    .option("multiline",true)
    .option("header", "true") // Use first line of all files as header
    .option("inferSchema", "true") // Automatically infer data types
    .option("escape", "@")
    .load("wasbs://uss@uosblob.blob.core.windows.net/Category.csv")

val USStraining = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("wholeFile", true)
    .option("multiline",true)
    .option("header", "true") // Use first line of all files as header
    .option("inferSchema", "true") // Automatically infer data types
    .option("escape", "@")

    .load("wasbs://uss@uosblob.blob.core.windows.net/USStraining.csv")

// COMMAND ----------


USStraining.write.saveAsTable("survey.USStraining")

// COMMAND ----------

// MAGIC %sql
// MAGIC select count(*) from survey.USStraining
// MAGIC --refresh table train

// COMMAND ----------

val sid_list = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("wholeFile", true)
    .option("multiline",true)
    .option("header", "true") // Use first line of all files as header
    .option("inferSchema", "true") // Automatically infer data types
    .option("escape", "@")
    .load("wasbs://uss@uosblob.blob.core.windows.net/sid_list.csv")

val ussresults = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("wholeFile", true)
    .option("multiline",true)
    .option("header", "true") // Use first line of all files as header
    .option("inferSchema", "true") // Automatically infer data types
    .option("escape", "@")
    .load("wasbs://uss@uosblob.blob.core.windows.net/ussresults.csv")

val sessiondates = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("wholeFile", true)
    .option("multiline",true)
    .option("header", "true") // Use first line of all files as header
    .option("inferSchema", "true") // Automatically infer data types
    .option("escape", "@")
    .load("wasbs://uss@uosblob.blob.core.windows.net/sessiondates.csv")

val ussquestionnaires = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("wholeFile", true)
    .option("multiline",true)
    .option("header", "true") // Use first line of all files as header
    .option("inferSchema", "true") // Automatically infer data types
    .option("escape", "@")
    .load("wasbs://uss@uosblob.blob.core.windows.net/ussquestionnaires.csv")

val uss_data = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("wholeFile", true)
    .option("multiline",true)
    .option("header", "true") // Use first line of all files as header
    .option("inferSchema", "true") // Automatically infer data types
    .option("escape", "@")
    .load("wasbs://uss@uosblob.blob.core.windows.net/uss_data.csv")

// COMMAND ----------

val uss_data = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("wholeFile", true)
    .option("multiline",true)
    .option("header", "true") // Use first line of all files as header
    .option("inferSchema", "true") // Automatically infer data types
    .option("escape", "\"")
    //.option("escape", "@")
    //.option("escape", "")
    .load("wasbs://uss@uosblob.blob.core.windows.net/uss_data.csv")

// COMMAND ----------

uss_data.write.saveAsTable("survey.uss_data_1")

// COMMAND ----------

//uss_data.write.saveAsTable("survey.uss_data")
sid_list.write.saveAsTable("survey.sid_list")
ussresults.write.saveAsTable("survey.ussresults")
sessiondates.write.saveAsTable("survey.sessiondates")
ussquestionnaires.write.saveAsTable("survey.ussquestionnaires")
//uss_data.write.saveAsTable("survey.uss_data")
//uss_data.write.saveAsTable("survey.uss_data")

// COMMAND ----------

display(uss_data)

// COMMAND ----------

val enr = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("wholeFile", true)
    .option("multiline",true)
    .option("header", "true") // Use first line of all files as header
    .option("inferSchema", "true") // Automatically infer data types
    .option("escape", "@")
    .load("wasbs://uss@uosblob.blob.core.windows.net/enr1.csv")

// COMMAND ----------

display(enr)

// COMMAND ----------

val enr1 = enr.withColumnRenamed("UOS occurrence","UOS_occurrence")
              .withColumnRenamed("salary","salary_amount")
              .withColumnRenamed("UOS occurrence","UOS_occurrence")
              .withColumnRenamed("UOS session","UOS_session")
              .withColumnRenamed("UOS level","UOS_level_des")
              .withColumnRenamed("uos_level","UOS_level_ID")


enr1.printSchema()

// COMMAND ----------

enr1.write.saveAsTable("survey.enr")

// COMMAND ----------

// MAGIC %sql
// MAGIC select * from survey.enr

// COMMAND ----------

