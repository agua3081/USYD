# Databricks notebook source
# MAGIC %md To ensure high quality in analytical modeling or analysis, data must be validated and cleansed. In our scenerio, we are given two sets of similar room type, one is sourced from Expedia, another is sourced from Booking.com. We will normalize both sets to have a common record. Fuzzy matching is a technique that I am using. It works with matches that may be less than 100% perfect. Fuzzy matching is blind to obvious synonyms.
# MAGIC 
# MAGIC In this exercise, I take room type from Expedia, compare and match it's associated room type in Booking.com. In another words, we match records between two data sources.
# MAGIC 
# MAGIC I have defined a match as something more like “a human with some experiences would have guessed these rooms were the same thing”. 

# COMMAND ----------

import pandas as pd


# COMMAND ----------

# MAGIC %sql
# MAGIC select row_number() over(ORDER BY text)-1 AS ID,text,parent,Category from survey.usstrainingView1level3

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from survey.ussdata20182019o1o2 where substring(time_stamp,1,4)=2019

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace view survey.trainingtestsetview as
# MAGIC select distinct a.text,a.parent,a.Category,b.text as btext,b.id,b.area from survey.usstrainingView1level3 a
# MAGIC left join survey.ussdata20182019o1o2 b on a.text=b.text

# COMMAND ----------

# MAGIC %sql
# MAGIC select text,parent from survey.trainingtestsetview where id is null

# COMMAND ----------

# MAGIC %md ### FuzzyWuzzy
# MAGIC 
# MAGIC Let's give a try, compare and match three pairs of the data.
# MAGIC 
# MAGIC 1). Ratio, - Compares the entire string similarity, in order.

# COMMAND ----------

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# COMMAND ----------

# MAGIC %scala
# MAGIC val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
# MAGIC val hiveDF = hiveContext.sql("select text,parent from survey.trainingtestsetview where id is null")
# MAGIC val hiveDF1 = hiveContext.sql("select * from survey.ussdata20182019o1o2 where substring(time_stamp,1,4)=2019")
# MAGIC hiveDF.createOrReplaceTempView("usstrainingView")
# MAGIC hiveDF1.createOrReplaceTempView("ussdataView")

# COMMAND ----------

articles_all = sqlContext.table("usstrainingView")
df1 = articles_all.toPandas()
articles_all1 = sqlContext.table("ussdataView")
df2 = articles_all1.toPandas()

# COMMAND ----------

df2.dropna()

# COMMAND ----------

df1

# COMMAND ----------

def match_name(name, list_names, min_score=0):

    # -1 score incase we don't get any matches
    max_score = -1
    # Returning empty name for no match as well
    max_name = ""
    # Iternating over all names in the other
    for name2 in list_names:
        #Finding fuzzy match score
        score = fuzz.ratio(name, name2)
        # Checking if we are above our threshold and have a better score
        if (score > min_score) & (score > max_score):
            max_name = name2
            max_score = score
    return (max_name, max_score)

# COMMAND ----------

# List for dicts for easy dataframe creation
dict_list = []
# iterating over our players without salaries found above
for text in df1.text:
    # Use our method to find best match, we can set a threshold here
    match = match_name(text, df2.text, 75)
    
    # New dict for storing data
    dict_ = {}
    dict_.update({"player_name" : text})
    dict_.update({"match_name" : match[0]})
    dict_.update({"score" : match[1]})
    dict_list.append(dict_)
    
merge_table = pd.DataFrame(dict_list)
# Display results
merge_table

# COMMAND ----------

word_stats = spark.createDataFrame(pd.DataFrame(merge_table), columns=['player_name','match_name','score'])
word_stats.createOrReplaceTempView('datasetpivot')

# COMMAND ----------

# MAGIC %scala
# MAGIC var wordStatsSpark = sqlContext.table("datasetpivot")
# MAGIC wordStatsSpark.write.mode("overwrite").saveAsTable("survey.FuzzyLookup")

# COMMAND ----------

def fw_process(row_df1):
    # Select the addresses from df2 with same postal_code
    #df2_select_add = df2['id'][df2['o1'] == row_df1['postal_code']]
    df2_select_add = [df2['text']]
    ad_1 = row_df1['text']
    # Find the best match for ad_1 in df2_select_add and get the ratio with [1] 
    # for the name of df2_select_add , use [0]
    if process.extractOne(ad_1, df2_select_add)[1] >= 80:
        return df2_select_add[0]
    else:
        return 'N'

# COMMAND ----------

ad_1['flag'] = ad_1.apply(fw_process , axis=1)

# COMMAND ----------

def fuzzy_match(x, choices, scorer, cutoff):
    match = process.extractOne(x['text'], 
                               choices=choices.loc[choices['Document_ID'] == x['Document_ID'], 'text'], 
                               scorer=scorer, 
                               score_cutoff=cutoff)
    if match:
        return match[0]

df1['FuzzyAddress1'] = df1.apply(fuzzy_match, 
                                   args=(df2, fuzz.ratio, 60), 
                                   axis=1)

# COMMAND ----------

df1

# COMMAND ----------

# MAGIC %md This is telling us that the "Deluxe Room, 1 King Bed" and "Deluxe King Room" pair are about 62% the same.

# COMMAND ----------

fuzz.ratio('Traditional Double Room, 2 Double Beds', 'Double Room with Two Double Beds')
fuzz.token_sort_ratio('Deluxe Room, 1 King Bed', 'Deluxe King Room')

# COMMAND ----------

# MAGIC %md The "Traditional Double Room, 2 Double Beds" and "Double Room with Two Double Beds" pair are about 69% the same.

# COMMAND ----------

fuzz.ratio('Room, 2 Double Beds (19th to 25th Floors)', 'Two Double Beds - Location Room (19th to 25th Floors)')

# COMMAND ----------

# MAGIC %md The "Room, 2 Double Beds (19th to 25th Floors)" and "Two Double Beds - Location Room (19th to 25th Floors)" pair are about 74% the same.

# COMMAND ----------

# MAGIC %md I am a little disappointed with these. It turns out, the naive approach is far too sensitive to minor differences in word order, missing or extra words, and other such issues.
# MAGIC 
# MAGIC 2). Partial ratio, - Compares partial string similarity.
# MAGIC 
# MAGIC We are still using the same data pairs.

# COMMAND ----------

fuzz.partial_ratio('Deluxe Room, 1 King Bed', 'Deluxe King Room')

# COMMAND ----------

fuzz.partial_ratio('Traditional Double Room, 2 Double Beds', 'Double Room with Two Double Beds')

# COMMAND ----------

fuzz.partial_ratio('Room, 2 Double Beds (19th to 25th Floors)', 'Two Double Beds - Location Room (19th to 25th Floors)')

# COMMAND ----------

# MAGIC %md Comparing partial string brings a little better results for some pairs.
# MAGIC 
# MAGIC 3). Token sort ratio, - Ignores word order.

# COMMAND ----------

fuzz.token_sort_ratio('Deluxe Room, 1 King Bed', 'Deluxe King Room')

# COMMAND ----------

fuzz.token_sort_ratio('Traditional Double Room, 2 Double Beds', 'Double Room with Two Double Beds')

# COMMAND ----------

fuzz.token_sort_ratio('Room, 2 Double Beds (19th to 25th Floors)', 'Two Double Beds - Location Room (19th to 25th Floors)')

# COMMAND ----------

# MAGIC %md The best so far.
# MAGIC 
# MAGIC 4). Token set ratio, - Ignores duplicated words. It is similar with token sort ratio, but a little bit more flexible.

# COMMAND ----------

fuzz.token_set_ratio('Deluxe Room, 1 King Bed', 'Deluxe King Room')

# COMMAND ----------

fuzz.token_set_ratio('Traditional Double Room, 2 Double Beds', 'Double Room with Two Double Beds')

# COMMAND ----------

fuzz.token_set_ratio('Room, 2 Double Beds (19th to 25th Floors)', 'Two Double Beds - Location Room (19th to 25th Floors)')

# COMMAND ----------

# MAGIC %md Seems token set ratio is the best fit for my data. According to this discovery, I decided to apply token set ratio to my entire data set.

# COMMAND ----------

# MAGIC %md When setting ratio > 70.

# COMMAND ----------

def get_ratio(row):
    name = row['Expedia']
    name1 = row['Booking.com']
    return fuzz.token_set_ratio(name, name1)

df[df.apply(get_ratio, axis=1) > 70].head(10)

# COMMAND ----------

len(df[df.apply(get_ratio, axis=1) > 70]) / len(df)

# COMMAND ----------

# MAGIC %md Over 90% of the pairs exceed a match score of 70.

# COMMAND ----------

df[df.apply(lambda row: fuzz.token_set_ratio(row['Expedia'], row['Booking.com']), axis=1) > 60]

# COMMAND ----------

len(df[df.apply(lambda row: fuzz.token_set_ratio(row['Expedia'], row['Booking.com']), axis=1) > 60]) / len(df)