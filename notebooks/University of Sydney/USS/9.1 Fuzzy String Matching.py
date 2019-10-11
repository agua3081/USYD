# Databricks notebook source
# MAGIC %md To ensure high quality in analytical modeling or analysis, data must be validated and cleansed. In our scenerio, we are given two sets of similar room type, one is sourced from Expedia, another is sourced from Booking.com. We will normalize both sets to have a common record. Fuzzy matching is a technique that I am using. It works with matches that may be less than 100% perfect. Fuzzy matching is blind to obvious synonyms.
# MAGIC 
# MAGIC In this exercise, I take room type from Expedia, compare and match it's associated room type in Booking.com. In another words, we match records between two data sources.
# MAGIC 
# MAGIC I have defined a match as something more like “a human with some experiences would have guessed these rooms were the same thing”. 

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install python_Levenshtein-0.12.0-cp35-none-win_amd64.whl

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install python-Levenshtein

# COMMAND ----------

import pandas as pd
import sys
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import datetime
import time
import Levenshtein
#df = pd.read_csv('room_type.csv')

# COMMAND ----------

#!/usr/bin/env python
import sys
from fuzzywuzzy import fuzz
import datetime
import time
import Levenshtein

# COMMAND ----------

#!/usr/bin/env python
import sys
from fuzzywuzzy import fuzz
import datetime
import time
import Levenshtein

#init for comparison
with open('normalized_set_record_set.csv') as normalized_records_ALL_file:
# with open('delete_this/xab') as normalized_records_ALL_file:
    normalized_records_ALL_dict = {}
    for line in normalized_records_ALL_file:
        key, value = line.strip('\n').split(':', 1)
        normalized_records_ALL_dict[key] = value
        # normalized_records_ALL_dict[contact_id] = concat_record

def score_it_bag(target_contact_id, target_str, ALL_records_dict):
    '''
    INPUT target_str, ALL_records_dict
    OUTPUT sorted list by highest fuzzy match
    '''
    return sorted([(value_str, contact_id_index_str, fuzz.ratio(target_str, value_str)) 
        for contact_id_index_str, value_str in ALL_records_dict.iteritems()], key=lambda x:x[2])[::-1]

def score_it_closest_match_pandas(target_contact_id, target_str, place_holder_delete):
    '''
    INPUT target_str, ALL_records_dict
    OUTPUT closest match
    '''
    # simply drop this index target_contact_id
    df_score = df_ALL.concat_record.apply(lambda x: fuzz.ratio(target_str, x))

    return df_ALL.concat_record[df_score.idxmax()], df_score.max(), df_score.idxmax()

def score_it_closest_match_L(target_contact_id, target_str, ALL_records_dict_input):
    '''
    INPUT target_str, ALL_records_dict
    OUTPUT closest match tuple (best matching str, score, contact_id of best match str)
    '''
    best_score = 100

    #score it
    for comparison_contactid, comparison_record_str in ALL_records_dict_input.iteritems():
        if target_contact_id != comparison_contactid:
            current_score = Levenshtein.distance(target_str, comparison_record_str)


            if current_score < best_score:
                best_score = current_score 
                best_match_id = comparison_contactid 
                best_match_str = comparison_record_str 

    return (best_match_str, best_score, best_match_id)



def score_it_closest_match_fuzz(target_contact_id, target_str, ALL_records_dict_input):
    '''
    INPUT target_str, ALL_records_dict
    OUTPUT closest match tuple (best matching str, score, contact_id of best match str)
    '''
    best_score = 0

    #score it
    for comparison_contactid, comparison_record_str in ALL_records_dict_input.iteritems():
        if target_contact_id != comparison_contactid:
            current_score = fuzz.ratio(target_str, comparison_record_str)

            if current_score > best_score:
                best_score = current_score 
                best_match_id = comparison_contactid 
                best_match_str = comparison_record_str 

    return (best_match_str, best_score, best_match_id)

def score_it_threshold_match(target_contact_id, target_str, ALL_records_dict_input):
    '''
    INPUT target_str, ALL_records_dict
    OUTPUT closest match tuple (best matching str, score, contact_id of best match str)
    '''
    score_threshold = 95

    #score it
    for comparison_contactid, comparison_record_str in ALL_records_dict_input.iteritems():
        if target_contact_id != comparison_contactid:
            current_score = fuzz.ratio(target_str, comparison_record_str)

            if current_score > score_threshold: 
                return (comparison_record_str, current_score, comparison_contactid)

    return (None, None, None)


def score_it_closest_match_threshold_bag(target_contact_id, target_str, ALL_records_dict):
    '''
    INPUT target_str, ALL_records_dict
    OUTPUT closest match
    '''
    threshold_score = 80
    top_matches_list = []
    #score it
    #iterate through dictionary
    for comparison_contactid, comparison_record_str in ALL_records_dict.iteritems():
        if target_contact_id != comparison_contactid:
            current_score = fuzz.ratio(target_str, comparison_record_str)

            if current_score > threshold_score:
                top_matches_list.append((comparison_record_str, current_score, comparison_contactid))


    if len(top_matches_list) > 0:  return top_matches_list

def score_it_closest_match_threshold_bag_print(target_contact_id, target_str, ALL_records_dict):
    '''
    INPUT target_str, ALL_records_dict
    OUTPUT closest match
    '''
    threshold_score = 80


    #iterate through dictionary
    for comparison_contactid, comparison_record_str in ALL_records_dict.iteritems():
        if target_contact_id != comparison_contactid:

            #score it
            current_score = fuzz.ratio(target_str, comparison_record_str)
            if current_score > threshold_score:
                print target_contact_id + ':' + str((target_str,comparison_record_str, current_score, comparison_contactid))


    pass


#stream in all contacts ie large set
for line in sys.stdin:
    # ERROR DIAG TOOL
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print >> sys.stderr, line, st

    contact_id, target_str = line.strip().split(':', 1)

    score_it_closest_match_threshold_bag_print(contact_id, target_str, normalized_records_ALL_dict)
    # output = (target_str, score_it_closest_match_fuzz(contact_id, target_str, normalized_records_ALL_dict))
    # output = (target_str, score_it_closest_match_threshold_bag(contact_id, target_str, normalized_records_ALL_dict))
    # print contact_id + ':' + str(output)


# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace view survey.ussdata2019o1o2 as
# MAGIC select row_number() over(ORDER BY ID)-1 AS Document_ID,ID,area,time_stamp,text from
# MAGIC (
# MAGIC select ID,time_stamp,o1 as text,'o1' as area  from survey.uss_data_1 where substring(time_stamp,1,4) in (2019) and o1 not in('null','N/A.','N/A','...','Nil','None.','None','none')
# MAGIC union
# MAGIC select ID,time_stamp,o2 as text,'o2' as area  from survey.uss_data_1 where substring(time_stamp,1,4) in (2019) and o2 not in('null','N/A.','N/A','...','Nil','None.','None','none')
# MAGIC )a

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
# MAGIC val hiveDF = hiveContext.sql("select text,parent from survey.usstrainingView")
# MAGIC val hiveDF1 = hiveContext.sql("select ID,o1,'o1' as area  from survey.uss_data_1 where substring(time_stamp,1,4) in (2018,2019) and o1 not in('null','N/A.','N/A','...','Nil','None.','None','none')")
# MAGIC hiveDF.createOrReplaceTempView("usstrainingView")
# MAGIC hiveDF1.createOrReplaceTempView("ussdataView")

# COMMAND ----------

articles_all = sqlContext.table("usstrainingView")
ad_1 = articles_all.toPandas()
articles_all1 = sqlContext.table("ussdataView")
df2 = articles_all1.toPandas()

# COMMAND ----------

df2.dropna()

# COMMAND ----------

def fw_process(row_df1):
    # Select the addresses from df2 with same postal_code
    #df2_select_add = df2['id'][df2['o1'] == row_df1['postal_code']]
    df2_select_add = df2['id'][df2['o1']]
    ad_1 = row_df1['text']
    # Find the best match for ad_1 in df2_select_add and get the ratio with [1] 
    # for the name of df2_select_add , use [0]
    if process.extractOne(ad_1, df2_select_add)[1] >= 80:
        return 'Y'
    else:
        return 'N'

# COMMAND ----------

ad_1['flag'] = ad_1.apply(fw_process , axis=1)

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