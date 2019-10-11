# Databricks notebook source
#%matplotlib inline
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score,f1_score,log_loss
#from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
#stop_words = stopwords.words('english')
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline
import seaborn as sns

# Run in python console
import nltk; nltk.download('stopwords')
from nltk.corpus import stopwords

# COMMAND ----------

# DBTITLE 1,Add Functions
# Add Functions
def error_measure(y_true, y_pred):
  
  metric = dbutils.widgets.get("error")

  # ensure the labels are binary
  y_true = y_true != 0
  y_pred = y_pred != 0
  
  if metric == "accuracy":
    error = accuracy_score(y_true, y_pred)
  if metric == "auc":
    error = roc_auc_score(y_true, y_pred)
  if metric == "precision":
    error = precision_score(y_true, y_pred)
  if metric == "F1 score":
    error = f1_score(y_true, y_pred)
  if metric == "log_loss":
    error = log_loss(y_true, y_pred)
    
  # round to 4 decimal places
  error = np.round(error, 4)
  
  return(error)

# score a given category with a certain pipeline
def get_category_error(pipeline, category, X_train, X_test, y_train, y_test):
    
    print('... Processing {}'.format(category))
    
    # train the model using X_dtm & y
    pipeline.fit(X_train, y_train)
    
    # compute the testing accuracy
    prediction = pipeline.predict(X_test)
    error = error_measure(y_test, prediction)
        
    print('Test accuracy is {}'.format(error))
    return error

# score a given model and pipeline across all categories
def get_model_error(model, categories, X_train, X_test, y_train, y_test):
  
    print('Model: {}'.format(model))
    pipeline = Pipeline([
                       ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                       ('clf', model),
                       ])
    
    for category in categories:
      category_error = get_category_error(pipeline, category, X_train, X_test, y_train[category], y_test[category])
      
      # write to results
      model_error = results.append(category_error)
    
    print('Column-wise mean test error: {}\n'.format(np.mean(results[-len(categories):])))
    return model_error

# score a given category by a pipeline (including a model)
def score_category_by_pipeline(pipeline, X_train, X_test, y_train, y_test):
      
    # y_train and y_test must already be subset to the category
    # only score the provided category
    #y_train = y_train[category]
    #y_test = y_test[category]
    
    #fit model
    pipeline.fit(X_train, y_train)
    
    # compute the testing accuracy
    prediction = pipeline.predict(X_test)
    error = error_measure(y_test, prediction)
    
    #print('{}: {}'.format(dbutils.widgets.get("error"), error))
    #print('Pipeline: {}'.format(pipeline.get_params()))
  
    return error

# clean up the text using regular expressions
def clean_text(text):
  text = text.lower()
  text = re.sub(r"what's", "what is ", text)
  text = re.sub(r"\'s", " ", text)
  text = re.sub(r"\'ve", " have ", text)
  text = re.sub(r"can't", "can not ", text)
  text = re.sub(r"n't", " not ", text)
  text = re.sub(r"i'm", "i am ", text)
  text = re.sub(r"\'re", " are ", text)
  text = re.sub(r"\'d", " would ", text)
  text = re.sub(r"\'ll", " will ", text)
  text = re.sub(r"\'scuse", " excuse ", text)
  text = re.sub('\W', ' ', text)
  text = re.sub('\s+', ' ', text)
  text = text.strip(' ')
  return text

# COMMAND ----------

# DBTITLE 1,Add Widgets
# add a widget to determine whether to stem after lemmatising
dbutils.widgets.dropdown("error", "accuracy", ["accuracy", "auc", "precision", "F1 score", "log_loss"], "error")

# TODO: add other models: "xgboost", "SVC"
dbutils.widgets.multiselect("models", "Logistic Regression", ["Logistic Regression", "LinearSVC", "Naive Bayes(MNB)", "SVC", "DecisionTree", "Random Forest", "xgboost", "extratrees", "Ridge"], "Models")

# get widget value
#metric = dbutils.widgets.get("error")

# remove widgets
#dbutils.widgets.removeAll()

# COMMAND ----------

# DBTITLE 1,Define constants, global parameters and candidate models
# Define random seed
seed = 42        # used for random state
test_ratio = 0.2 # size of validation set
folds = 1        # number of folds to use

# Define candidate models
candidate_models = {
                    'Logistic Regression': LogisticRegression(solver='sag', random_state = seed, multi_class = 'ovr'),
                    'LinearSVC': LinearSVC(random_state = seed),
                    'Naive Bayes(MNB)': MultinomialNB(fit_prior=True, class_prior=None),
                    'SVC': SVC(probability = True, gamma = 'scale'),
                    'DecisionTree': DecisionTreeClassifier(random_state=seed),
                    'Random Forest': RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = seed),
                    'xgboost': xgb.XGBClassifier(objective="reg:logistic", random_state = seed),
                    'extratrees': ExtraTreesClassifier(n_estimators = 1000, random_state = seed, n_jobs = -1),
                    'Ridge': RidgeClassifier(random_state = seed)                 
                   }

# COMMAND ----------

# MAGIC %sql
# MAGIC --refresh survey.usstrainingPivot
# MAGIC select * from survey.usstrainingPivot

# COMMAND ----------

# MAGIC %scala
# MAGIC val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
# MAGIC val hiveDF = hiveContext.sql("select * from survey.usstrainingPivot")
# MAGIC //val personnelTable = spark.catalog.getTable("survey.usstrainingView")
# MAGIC hiveDF.createOrReplaceTempView("usstrainingView")

# COMMAND ----------

df = sqlContext.table("usstrainingView")
df = df.toPandas()

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md ### Number of comments in each category

# COMMAND ----------

df_toxic = df.drop(['text'], axis=1)
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

# MAGIC %md ### Multi-Label
# MAGIC 
# MAGIC How many comments have multiple labels?

# COMMAND ----------

import seaborn as sns
rowsums = df.iloc[:,1:].sum(axis=1)
x=rowsums.value_counts()

#plot
plt.figure(figsize=(8,5))
ax = sns.barplot(x.index, x.values)
plt.title("Multiple categories per comment")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('# of categories', fontsize=12)
display()

# COMMAND ----------

# MAGIC %md Vast majority of the comment texts are not labeled.

# COMMAND ----------

# MAGIC %md The distribution of the number of words in comment texts.

# COMMAND ----------

lens = df.text.str.len()
lens.hist(bins = np.arange(0,5000,50))

# COMMAND ----------

# MAGIC %md Most of the comment text length are within 500 characters, with some outliers up to 5,000 characters long.

# COMMAND ----------

print('Percentage of comments that are not labelled:')
print(len(df[(df['AD']==0) & (df['CU']==0) & (df['GQ']==0) & (df['LC']== 0) & (df['LE']==0) & (df['LR']==0) & (df['SU']==0) & (df['TQ']==0)]) / len(df))

# COMMAND ----------

# MAGIC %md There is no missing comment in comment text column.

# COMMAND ----------

print('Number of missing comments in comment text:')
df['text'].isnull().sum()

# COMMAND ----------

# MAGIC %md Have a peek the first comment, the text needs clean.

# COMMAND ----------

df['text'][0]

# COMMAND ----------

categories = ['AD','CU','GQ','LC','LE','LR','SU','TQ']
num_of_categories = len(categories)

# COMMAND ----------

# MAGIC %md ### Clean up comment_text column 

# COMMAND ----------

df['text'] = df['text'].map(lambda com : clean_text(com))

# COMMAND ----------

df['text'][0]

# COMMAND ----------

# MAGIC %md ### Pre-processing

# COMMAND ----------

# NLTK Stop words
stop_words = stopwords.words('english')
stop_words.extend(['fuck','not','do','be','go','s','take','make','want','many','much','get','xx','nan','even','anyone','still','guy','already','maybe','honestly','man','dud','mate','actually','stuff','mostly','usyd', 'university','actual','realli','onli','rants','someon','everyon','anyon', 'ive', 'uni','rant','univers','everyone','someone','am','pm','http','https','www','day','today','week','month','year','new'])

# COMMAND ----------

# MAGIC %md ### Pipeline
# MAGIC 
# MAGIC scikit-learn provides a Pipeline utility to help automate machine learning workflows. Pipelines are very common in Machine Learning systems, since there is a lot of data to manipulate and many data transformations to apply. So we will utilize pipeline to train every classifier.

# COMMAND ----------

# MAGIC %md ### Model Training per category

# COMMAND ----------

# # Determine models to use
model_names = dbutils.widgets.get("models").split(",")
models = [candidate_models[x] for x in model_names]
print(model_names)

# COMMAND ----------

# Use Stratified Shuffle Split for each model
# define X and y
X = df['text']
y = df[categories]

# clean up values to only be binary values
for column in y:
  y[column] = y[column] != 0
  
# define stratified train/test splits to maintain the class distribution across train and test sets
sss = StratifiedShuffleSplit(n_splits=folds, test_size=test_ratio, random_state=seed)
results = []
for model in models:
  
  print('\n{}'.format(model))
  # define the Pipeline
  pipeline = Pipeline([
                     ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                     ('clf', model),
                     ])
  for category in categories:
    for train_index, test_index in sss.split(X, y[category]):
      #print(train_index, test_index)
      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
      y_train, y_test = y[category].iloc[train_index], y[category].iloc[test_index]
      
      error = score_category_by_pipeline(pipeline, X_train, X_test, y_train, y_test)
      print('{}: {}={}'.format(category, dbutils.widgets.get("error"), error))
      results.append(error)

# COMMAND ----------

# summary results for model first
for id, model in enumerate(models):
  print("Mean Columnwise error: {}".format(np.mean(results[(id*num_of_categories):((id +1 ) * num_of_categories)])))
  print("{}\n".format(model))

# COMMAND ----------

# TODO: Write results and params to a database table

# COMMAND ----------

# Hyperparameter tuning of xgboost


# COMMAND ----------

# # define X and y
# X = df['text']
# y = df[categories]

# # clean up values to only be binary
# for column in y:
#   y[column] = y[column] != 0

# # define stratified train/test splits 
# sss = StratifiedShuffleSplit(n_splits=folds, test_size=test_ratio, random_state=seed)

# for category in categories:
#   for train_index, test_index in sss.split(X, y[category]):
#     print(train_index, test_index)
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
      
#     # run training here
#     results = []
#     for model in models:
#       get_model_error(model, categories, X_train, X_test, y_train, y_test)

# COMMAND ----------

# model, folds, results