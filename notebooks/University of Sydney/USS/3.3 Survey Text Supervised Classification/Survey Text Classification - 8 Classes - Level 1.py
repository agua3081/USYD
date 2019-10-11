# Databricks notebook source
# MAGIC %md
# MAGIC ## Introduction 
# MAGIC Select responses across Unit of Study areas that align to current University focus areas i.e. Experiential Learning currently.
# MAGIC Survey text records include responses for following two questions :<br><br>
# MAGIC <b>a) What are the best aspects of your experience </b> <br>
# MAGIC <b>b) What can be improved?</b> 
# MAGIC <br>
# MAGIC <br>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Utility Functions

# COMMAND ----------

# Python functions for Statistical Learning
# Author: Marcel Scharth, The University of Sydney Business School
# This version: 31/10/2017

# Imports
from pandas.core import datetools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import itertools


from sklearn.model_selection import cross_val_predict, LeaveOneOut
from statsmodels.nonparametric.kernel_regression import KernelReg



def plot_histogram(series):
    fig, ax= plt.subplots(figsize=(9,6))
    sns.distplot(series, ax=ax, hist_kws={'alpha': 0.9, 'edgecolor':'black'},  
        kde_kws={'color': 'black', 'alpha': 0.7})
    sns.despine()
    return fig, ax


def plot_histograms(X):

    labels = list(X.columns)
    
    N, p = X.shape

    rows = int(np.ceil(p/3)) 

    fig, axes = plt.subplots(rows, 3, figsize=(12, rows*(12/4)))

    for i, ax in enumerate(fig.axes):
        if i < p:
            sns.distplot(X.iloc[:,i], ax=ax, hist_kws={'alpha': 0.9, 'edgecolor':'black'},  
                kde_kws={'color': 'black', 'alpha': 0.7})
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title(labels[i])
            ax.set_yticks([])
            ax.set_xticks([])
        else:
            fig.delaxes(ax)

    sns.despine()
    plt.tight_layout()
    
    return fig, axes


def plot_correlation_matrix(X):

    fig, ax = plt.subplots()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(X.corr(), ax=ax, cmap=cmap)
    ax.set_title('Correlation matrix', fontweight='bold', fontsize=13)
    plt.tight_layout()

    return fig, ax



def plot_logistic_regressions(X, y):
    labels = list(X.columns)
    
    N, p = X.shape

    rows = int(np.ceil(p/3)) 

    fig, axes = plt.subplots(rows, 3, figsize=(12, rows*(11/4)))

    for i, ax in enumerate(fig.axes):
        if i < p:
            ax.set_xlim(auto=True)
            sns.regplot(X.iloc[:,i], y,  ci=None, logistic=True, y_jitter=0.05, 
                        scatter_kws={'s': 25, 'alpha':.5}, ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(labels[i])
        else:
            fig.delaxes(ax)

    sns.despine()
    plt.tight_layout()

# This function is from the scikit-learn documentation
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

import seaborn as sns

def plot_feature_importance(model, labels, max_features = 20):
    feature_importance = model.feature_importances_*100
    feature_importance = 100*(feature_importance/np.max(feature_importance))
    table = pd.Series(feature_importance, index = labels).sort_values(ascending=True, inplace=False)
    fig, ax = fig, ax = plt.subplots(figsize=(9,6))
    if len(table) > max_features:
        table.iloc[-max_features:].T.plot(kind='barh', edgecolor='black', width=0.7, linewidth=.8, alpha=0.9, ax=ax)
    else:
        table.T.plot(kind='barh', edgecolor='black', width=0.7, linewidth=.8, alpha=0.9, ax=ax)
    ax.tick_params(axis=u'y', length=0) 
    ax.set_title('Variable importance', fontsize=13)
    sns.despine()
    return fig, ax



from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def plot_roc_curves(y_test, y_probs, labels, sample_weight=None):
    
    fig, ax= plt.subplots(figsize=(9,6))

    N, M=  y_probs.shape

    for i in range(M):
        fpr, tpr, _ = roc_curve(y_test, y_probs[:,i], sample_weight=sample_weight)
        auc = roc_auc_score(y_test, y_probs[:,i], sample_weight=sample_weight)
        ax.plot(1-fpr, tpr, label=labels.iloc[i] + ' (AUC = {:.3f})'.format(auc))
    
    ax.plot([0,1],[1,0], linestyle='--', color='black', alpha=0.6)

    ax.set_xlabel('Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title('ROC curves', fontsize=14)
    sns.despine()

    plt.legend(fontsize=13, loc ='lower left' )
    
    return fig, ax

def bootstrap_mean(y, S=1000, alpha=0.05):
    y = np.ravel(y)
    N = len(y) # It is useful to store the size of the data

    mean_boot=np.zeros(S)
    t_boot=np.zeros(S)

    y_mean = np.mean(y)
    se = np.std(y, ddof=1)/np.sqrt(N)

    for i in range(S):
        y_boot = y[np.random.randint(N, size=N)] 
        mean_boot[i] = np.mean(y_boot)
        se_boot = np.std(y_boot, ddof=1)/np.sqrt(N)
        t_boot[i]=(mean_boot[i]-y_mean)/se_boot

    ci_low =  y_mean-se*np.percentile(t_boot, 100*(1-alpha/2))
    ci_high = y_mean-se*np.percentile(t_boot, 100*(alpha/2))

    return mean_boot, ci_low, ci_high

# COMMAND ----------

#nltk.download('wordnet')
#nltk.download('stopword')

# COMMAND ----------

# MAGIC %md
# MAGIC ##Import Libraries

# COMMAND ----------

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import spacy
import lightgbm as lgb
import gensim
import nltk
import re

from wordcloud import WordCloud
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,LancasterStemmer,SnowballStemmer,WordNetLemmatizer
from nltk.probability import FreqDist
from nltk import word_tokenize 
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import RidgeClassifierCV,RidgeClassifier,LogisticRegressionCV,LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from scipy.stats import randint
from sklearn.metrics.pairwise import cosine_similarity
from gensim import models
warnings.filterwarnings("ignore")


# COMMAND ----------

# MAGIC %md
# MAGIC ##Settings

# COMMAND ----------

palette = ['#1F77B4', '#FF7F0E', '#2CA02C', '#DB2728', '#9467BD', '#8C564B', '#E377C2','#7F7F7F', '#BCBD22', '#17BECF','#67E568','#257F27','#08420D','#FFF000','#FFB62B','#E56124','#E53E30','#7F2353','#F911FF','#9F8CA6']
pd.set_option('display.max_colwidth', -1)
sns.set_context('notebook') 
sns.set_style('ticks')

# COMMAND ----------

labels = ['Administration','Curriculum','Graduate Qualities','Learning Community/ Belonging','Learner Engagement','Learning Resources','Quality of Supervision','Teaching Quality']
df_columns = ['AD','CU','GQ','LC','LE','LR','SU','TQ']
holdout_no=3000

# COMMAND ----------

# MAGIC %md
# MAGIC #Read Data

# COMMAND ----------

# MAGIC %scala
# MAGIC val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
# MAGIC val hiveDF = hiveContext.sql("select * from survey.usstrainingPivot")
# MAGIC //val personnelTable = spark.catalog.getTable("survey.usstrainingView")
# MAGIC hiveDF.createOrReplaceTempView("usstrainingView")

# COMMAND ----------

df_raw = sqlContext.table("usstrainingView")
df_raw = df_raw.toPandas()

# COMMAND ----------

#df = pd.read_csv('Datasets/usyd-survey.csv')
df_raw.head(5)


# COMMAND ----------

# MAGIC %md
# MAGIC ##Train and Test (holdout) sets

# COMMAND ----------

df=df_raw
holdout_index = df.sample(n=holdout_no, random_state=42).index
df_holdout = df.loc[holdout_index]
df_holdout.reset_index(drop=True,inplace=True)
df.reset_index(drop=True,inplace=True)
df = df.drop(df.sample(n=holdout_no).index)
df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #Exploratory Data Analysis (EDA)

# COMMAND ----------

# MAGIC %md
# MAGIC Find and drop any unlabelled records

# COMMAND ----------

rowsums = df.iloc[:,1:].sum(axis=1)
x = rowsums.value_counts()
x

# COMMAND ----------

df

# COMMAND ----------


plt.figure(figsize=(19,15))
pd.DataFrame(df.iloc[:,1:].sum(axis=0)).transpose()
g=sns.barplot(data=pd.DataFrame(df.iloc[:,1:].sum(axis=0)).transpose().sort_values(by=0,axis=1,ascending=False, inplace=False))
a=g.set_xticklabels(g.get_xticklabels(),rotation=35)
display()

# COMMAND ----------

df.drop(df[df.iloc[:,1:].sum(axis=1)==0].index,inplace=True)
df.info()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Text Cleaning/Stemming/Lemmatizing

# COMMAND ----------

data = df.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
stop_words = stopwords.words('english')
stop_words.extend(
['not','do','be','go','s','take','make','want','many','much','get','xx','nan','even','anyone','still','guy','already','maybe','honestly','man','dud',
 'mate','actually','stuff','mostly','usyd', 'university','actual','realli','onli','rants','someon','everyon','anyon', 'ive', 'uni','rant','univers',
 'everyone','someone','am','pm','http','https','www','day','today','week','month','year','new'])


def clean_text(text):
    """
        text: a string       
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
#    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words) # remove stopwors from text
    return text

    
def porter_stemmer(text):
    """
        text: a string       
        return: stemmed string
    """    
    ps = PorterStemmer() 
    text = text.lower()
    text = ' '.join(ps.stem(word) for word in text.split())
    return text

def lancaster_stemmer(text):
    """
        text: a string       
        return: stemmed string
    """    
    ls = LancasterStemmer() 
    text = text.lower()
    text = ' '.join(ls.stem(word) for word in text.split())
    return text

def snowball_stemmer(text):
    """
        text: a string       
        return: stemmed string
    """    
    ss = SnowballStemmer() 
    text = text.lower()
    text = ' '.join(ss.stem(word) for word in text.split())
    return text

def wordnet_lemmatize(text):
    """
        text: a string       
        return: lemmatize string
    """    
    wl = WordNetLemmatizer()
    text = text.lower()
    text = ' '.join(wl.lemmatize(word,pos="v") for word in text.split())
    return text


def spacy_lemmatize_pos(text):
    """
        text: a string       
        return: lemmatize string
    """    

    nlp = spacy.load('en_core_web_sm')
    lemma = []
    pos = []
    allowed_postags= ['NOUN', 'ADJ', 'PROPN','VERB']
    for doc in nlp.pipe(text.astype('unicode').values, batch_size=50,n_threads=-1):
        lemma.append(" ".join(str(v) for v in [n.lemma_ for n in doc if n.pos_ in allowed_postags]) )
        pos.append([n.pos_ for n in doc])
    df['postext'] = pos
    return pd.DataFrame(lemma)

df['clntext'] = df['text'].apply(clean_text)
df['clntext'] = df['clntext'].str.replace('\d+', '')
#df['clntext'] = df['clntext'].apply(porter_stemmer)
#df['stmtext'] = df['clntext'].apply(lancaster_stemmer)
df['stmlemtext'] = df['clntext'].apply(wordnet_lemmatize)

#df['stmlemtext'] = spacy_lemmatize_pos(df['clntext'])

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Down Sampling Data

# COMMAND ----------

# from sklearn.utils import resample

# # downsample majority
# new_size = 2500
# TQ_downsampled = resample(df[df['TQ']==1],
#                                 replace = False, # sample without replacement
#                                 n_samples = new_size, # match minority n
#                                 random_state = 27) # reproducible results
# CU_downsampled = resample(df[df['CU']==1],
#                                 replace = False, # sample without replacement
#                                 n_samples = new_size, # match minority n
#                                 random_state = 27) # reproducible results
# GQ_downsampled = resample(df[df['GQ']==1],
#                                 replace = False, # sample without replacement
#                                 n_samples = new_size, # match minority n
#                                 random_state = 27) # reproducible results
# # combine minority and downsampled majority
# downsampled = pd.concat([TQ_downsampled,CU_downsampled,GQ_downsampled,
#                           df[df['LE']==1] ,df[df['LR']==1],df[df['AD']==1]
#                         , df[df['LC']==1], df[df['SU']==1]
#                         ])

# # data = downsampled
# plt.rcParams['figure.figsize'] = (9, 6)
# pd.DataFrame(downsampled.iloc[:,1:].sum(axis=0)).transpose()
# sns.barplot(data=pd.DataFrame(downsampled.iloc[:,1:9].sum(axis=0)).transpose().sort_values(by=0,axis=1,
#                              ascending=False, inplace=False))

# COMMAND ----------

data=df.copy()

# COMMAND ----------

#pd.set_option('display.max_colwidth', -1)
data['text'].head(2)

# COMMAND ----------

mask=data['text'].str.len()<20
df_w = data.loc[mask]
df_w.head(10)

# COMMAND ----------

len(df_w)

# COMMAND ----------

data=data.sample(frac=1)
data.reset_index(drop=True,inplace=True)
data.head(2)

# COMMAND ----------

max(data['stmlemtext'], key=len)

# COMMAND ----------

min(data['stmlemtext'], key=len)

# COMMAND ----------

mask=data['stmlemtext'].str.len()<3
df_w = data.loc[mask]
df_w.head(1000)



# COMMAND ----------

data.drop(data[data['stmlemtext'].str.len()<3].index,inplace=True)
mask=data['stmlemtext'].str.len()<3
df_w = data.loc[mask]
df_w.head(1000)


data.drop(data[data['stmlemtext'].str.len()<3].index,inplace=True)

# COMMAND ----------

data['classes'] = data[df_columns].apply(lambda x: x.idxmax(), axis = 1)
data.head(5)

# COMMAND ----------



# COMMAND ----------

df_holdout['classes'] = df_holdout[df_columns].apply(lambda x: x.idxmax(), axis = 1)
df_holdout.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tokenize the text

# COMMAND ----------

# Create a list of tokens for each sentence
tokenizer = RegexpTokenizer(r'\w+')
data["tokens"] = data["stmlemtext"].apply(tokenizer.tokenize)

all_words = [word for tokens in data["tokens"] for word in tokens]
sentence_lengths = [len(tokens) for tokens in data["tokens"]]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
print("Max survey text length is %s" % max(sentence_lengths))

# COMMAND ----------

data.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## World Cloud

# COMMAND ----------

words=data[data['TQ']==1]['stmlemtext']

fig, ax = plt.subplots(figsize=(12,9))
wordcloud = WordCloud(background_color="white", collocations=False).generate(str(words))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
display()

# COMMAND ----------

words=data[data['AD']==1]['stmlemtext']

fig, ax = plt.subplots(figsize=(12,9))
wordcloud = WordCloud(background_color="white", collocations=False).generate(str(words))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show
display()

# COMMAND ----------

words=data[data['LC']==1]['stmlemtext']

fig, ax = plt.subplots(figsize=(12,9))
wordcloud = WordCloud(background_color="white", collocations=False).generate(str(words))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
display()


# COMMAND ----------

words=data[data['GQ']==1]['stmlemtext']

fig, ax = plt.subplots(figsize=(12,9))
wordcloud = WordCloud(background_color="white", collocations=False).generate(str(words))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

display()

# COMMAND ----------

fig = plt.figure(figsize=(12, 10)) 
plt.title('Sentence length histogram')
plt.xlabel('survey text token length')
plt.ylabel('number of survey tokens')
plt.hist(sentence_lengths, edgecolor = 'black', bins = 30)
plt.show()
display()

# COMMAND ----------

print(f"Median survey text token length {np.median(sentence_lengths)}")
print(f"Mean survey text token length {round(np.mean(sentence_lengths), 2)}")

# COMMAND ----------


Y = data[df_columns]
X_train, X_test, Y_train, Y_test = train_test_split(data['stmlemtext'],Y, test_size = 0.1, random_state = 42)

tvec = TfidfVectorizer(max_features=100000,ngram_range=(1, 3))
X_train_tfidf = tvec.fit_transform(X_train)
X_validation_tfidf = tvec.transform(X_test)
chi2score = chi2(X_train_tfidf, Y_train)[0]


plt.figure(figsize=(15,10))
wscores = zip(tvec.get_feature_names(), chi2score)
wchi2 = sorted(wscores, key=lambda x:x[1])
topchi2 = tuple(zip(*wchi2[-50:]))
x = range(len(topchi2[1]))
labels = topchi2[0]
plt.barh(x,topchi2[1], align='center', alpha=0.2)
plt.plot(topchi2[1], x, '-o', markersize=5, alpha=0.8)
plt.yticks(x, labels)
plt.xlabel('$\chi^2$')
display()

# COMMAND ----------


lst=[]
lst.append([' '.join(pd.DataFrame(data).query("AD == 1")['stmlemtext'])][0])
lst.append([' '.join(pd.DataFrame(data).query("CU == 1")['stmlemtext'])][0])
lst.append([' '.join(pd.DataFrame(data).query("GQ == 1")['stmlemtext'])][0])
lst.append([' '.join(pd.DataFrame(data).query("LC == 1")['stmlemtext'])][0])
lst.append([' '.join(pd.DataFrame(data).query("LE == 1")['stmlemtext'])][0])
lst.append([' '.join(pd.DataFrame(data).query("LR == 1")['stmlemtext'])][0])
lst.append([' '.join(pd.DataFrame(data).query("SU == 1")['stmlemtext'])][0])
lst.append([' '.join(pd.DataFrame(data).query("TQ == 1")['stmlemtext'])][0])

# Create the Document Term Matrix
count_vectorizer = CountVectorizer(stop_words='english')
count_vectorizer = CountVectorizer()
sparse_matrix = count_vectorizer.fit_transform(lst)

# OPTIONAL: Convert Sparse Matrix to Pandas Dataframe if you want to see the word frequencies.
doc_term_matrix = sparse_matrix.todense()
df_all = pd.DataFrame(doc_term_matrix, 
                  columns=count_vectorizer.get_feature_names(), 
                  index=['AD','CU','GQ','LC','LE','LR','SU','TQ'])
df_all

# COMMAND ----------

sim_df = pd.DataFrame(cosine_similarity(df_all, df_all))
sim_df=sim_df.rename(columns={0: df_columns[0],1: df_columns[1],2: df_columns[2],
                       3: df_columns[3],4: df_columns[4],5: df_columns[5],
                       6: df_columns[6],7: df_columns[7]})
sim_df.index =df_columns

# COMMAND ----------

sim_df

# COMMAND ----------

colormap = plt.cm.RdPu
plt.figure(figsize=(14,12))
corr01 = sim_df
plt.title('Cosine Similarity Matrix', y=1.05, size=15)
sns.heatmap(corr01.astype(float),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)

display()

# COMMAND ----------


# freq_list = sum(data['stmlemtext'].map(word_tokenize), [])
# word_frequencies = FreqDist(freq_list)
# word_frequencies

# COMMAND ----------

# freq_df = pd.DataFrame.from_dict(word_frequencies,orient='index').sort_values(0).reset_index()
# freq_df.rename(columns={"index": "word", 0: "frequency"},inplace=True)
# freq_df[freq_df['frequency']== 1].head(10)

# COMMAND ----------

# freq_df[freq_df['frequency']>= 2000]

# COMMAND ----------

# high_freq_list = list(freq_df[freq_df['frequency'] >= 2000]['word'])
# low_freq_list = list(freq_df[freq_df['frequency']== 1]['word'])
# def remove_low_high_freq(text):
#     """
#         text: a string       
#         return: modified initial string
#     """
#     text = text.lower() # lowercase text
#     text = REPLACE_BY_SPACE_RE.sub(' ', text)
#     text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
#     text = ' '.join(word for word in text.split() if word not in low_freq_list) # remove stopwors from text
#     text = ' '.join(word for word in text.split() if word not in high_freq_list) # remove stopwors from text
#     return text

# #data["stmlemtext"]=data["stmlemtext"].apply(remove_low_high_freq)

# COMMAND ----------

# freq_df[freq_df['frequency'] >= 1500].sort_values('frequency' , ascending=False)

# COMMAND ----------

# freq_list = sum(data['stmlemtext'].map(word_tokenize), [])
# word_frequencies = FreqDist(freq_list)
# word_frequencies

# freq_df = pd.DataFrame.from_dict(word_frequencies,orient='index').sort_values(0).reset_index()
# freq_df.rename(columns={"index": "word", 0: "frequency"},inplace=True)

# freq_df[freq_df['frequency']== 1].head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Development

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training Functions

# COMMAND ----------


# pretrained_embeddings_path = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
# word2vec =gensim.models.KeyedVectors.load_word2vec_format(pretrained_embeddings_path,binary=True)
#word2vec = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True,limit=10000)

# COMMAND ----------


def cntvectorizer(data):   
    df_temp = data.copy(deep = True)
    count_vectorizer = CountVectorizer(stop_words=stop_words, binary = True) # binary = True (could be better for logistic regression than non-binary, unless the counts are normalised first) 
    count_vectorizer.fit(df_temp['stmlemtext'])
    list_corpus = df_temp["stmlemtext"].tolist()       
    X = count_vectorizer.transform(list_corpus)   
    return X

def tfidf(data, ngrams = 1):
    df_temp = data.copy(deep = True)    
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, ngrams),stop_words=stop_words)
    tfidf_vectorizer.fit(df_temp['stmlemtext'])
    list_corpus = df_temp["stmlemtext"].tolist()
    X = tfidf_vectorizer.transform(list_corpus) 
    return X

# def w2v(data):    
#     df_temp = data.copy(deep = True)        
#     embeddings = get_word2vec_embeddings(Word2Vec, df_temp)   
#     return embeddings
  
# def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
#     if len(tokens_list)<1:
#         return np.zeros(k)
#     if generate_missing:
#         vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
#     else:
#         vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
#     length = len(vectorized)
#     summed = np.sum(vectorized, axis=0)
#     averaged = np.divide(summed, length)
#     return averaged

# def get_word2vec_embeddings(vectors, clean_text, generate_missing=False):
#     embeddings = clean_text['tokens'].apply(lambda x: get_average_word2vec(x, vectors,generate_missing=generate_missing))
#     return list(embeddings)


def get_metrics(y_test, y_predicted):  
    precision = precision_score(y_test, y_predicted, average='weighted')             
    recall = recall_score(y_test, y_predicted, average='weighted')   
    f1 = f1_score(y_test, y_predicted, average='weighted')
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1
  
  
  
  
def training_logreg(X_train_log, X_test_log, y_train_log, y_test_log, preproc):
    
    folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)   
    clf = LogisticRegressionCV(cv = folds, solver = 'saga', multi_class = 'multinomial', n_jobs = -1, penalty='l2')   
    clf.fit(X_train_log, y_train_log)

    res = pd.DataFrame(columns = ['Preprocessing', 'Model', 'Precision', 'Recall', 'F1-score', 'Accuracy'])
    
    y_pred = clf.predict(X_test_log)
    
    f1 = f1_score(y_pred, y_test_log, average = 'weighted')
    pres = precision_score(y_pred, y_test_log, average = 'weighted')
    rec = recall_score(y_pred, y_test_log, average = 'weighted')
    acc = accuracy_score(y_pred, y_test_log)
    
    res = res.append({'Preprocessing': preproc, 'Model': f'Logistic Regression', 'Precision': pres, 
                     'Recall': rec, 'F1-score': f1, 'Accuracy': acc}, ignore_index = True)

#     params = {
#                      'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,
#                      'class_weight': [ 'balanced',None] ,
#                      'multi_class': ['ovr', 'multinomial'] ,
#                      'penalty' : [  'l2' ],
#                      'solver' : ['saga'],
#                      'l1_ratio' : np.arange(0.0, 1.0, 0.1)
#              }
    
#     param_comb = 2
#     log = LogisticRegression()
#     skf = StratifiedKFold(n_splits=nfolds, shuffle = True, random_state = 1001)
#     random_search = RandomizedSearchCV(log , param_distributions=params, n_iter=param_comb, 
#                                        n_jobs=-1, cv=skf.split(X_train_log,y_train_log), verbose=3, random_state=1001 )
        
#     random_search.fit(X_train_log, y_train_log)
#     best_param = random_search.best_params_
#     print(best_param)
    
#     log = LogisticRegression(C=best_param['C'],class_weight=best_param['class_weight'],multi_class=best_param['multi_class']
#                              ,penalty=best_param['penalty'],solver=best_param['solver'],l1_ratio=best_param['l1_ratio']).fit(X_train_log, y_train_log)
    
#     y_pred = log.predict(X_test_log)
    
#     res = pd.DataFrame(columns = ['Preprocessing', 'Model', 'Precision', 'Recall', 'F1-score', 'Accuracy'])

#     f1 = f1_score(y_pred, y_test_log, average = 'weighted')
#     pres = precision_score(y_pred, y_test_log, average = 'weighted')
#     rec = recall_score(y_pred, y_test_log, average = 'weighted')
#     acc = accuracy_score(y_pred, y_test_log)
    
#     res = res.append({'Preprocessing': preproc, 'Model': 'Logistic Regression', 'Precision': pres, 
#                      'Recall': rec, 'F1-score': f1, 'Accuracy': acc}, ignore_index = True)

    return res


def training_naive(X_train_naive, X_test_naive, y_train_naive, y_test_naive, preproc):
    
    clf = MultinomialNB()
    clf.fit(X_train_naive, y_train_naive)

    res = pd.DataFrame(columns = ['Preprocessing', 'Model', 'Precision', 'Recall', 'F1-score', 'Accuracy'])
    
    y_pred = clf.predict(X_test_naive)
    
    f1 = f1_score(y_pred, y_test_naive, average = 'weighted')
    pres = precision_score(y_pred, y_test_naive, average = 'weighted')
    rec = recall_score(y_pred, y_test_naive, average = 'weighted')
    acc = accuracy_score(y_pred, y_test_naive)
    
    res = res.append({'Preprocessing': preproc, 'Model': 'Naive Bayes', 'Precision': pres, 
                     'Recall': rec, 'F1-score': f1, 'Accuracy': acc}, ignore_index = True)

    return res
  
  
def training_svc(X_train_svc, X_test_svc, y_train_svc, y_test_svc, nfolds , preproc):

    params = {
                     'C': [0.001, 0.01, 0.1, 1, 10, 100],
                     'gamma' : [0.001, 0.01, 0.1, 1]
             }
    param_comb = 2
    svc = svm.SVC(kernel='rbf')
    skf = StratifiedKFold(n_splits=nfolds, shuffle = True, random_state = 1001)
    random_search = RandomizedSearchCV(svc, param_distributions=params, n_iter=param_comb, 
                                     n_jobs=-1, cv=skf.split(X_train_svc,y_train_svc), verbose=3, random_state=1001 )
        
    random_search.fit(X_train_svc, y_train_svc)
    best_param = random_search.best_params_
    
    svc = svm.SVC(kernel='rbf', C=best_param["C"], gamma=best_param["gamma"]).fit(X_train_svc, y_train_svc)
    y_pred = svc.predict(X_test_svc)
    
    res = pd.DataFrame(columns = ['Preprocessing', 'Model', 'Precision', 'Recall', 'F1-score', 'Accuracy'])

    f1 = f1_score(y_pred, y_test_svc, average = 'weighted')
    pres = precision_score(y_pred, y_test_svc, average = 'weighted')
    rec = recall_score(y_pred, y_test_svc, average = 'weighted')
    acc = accuracy_score(y_pred, y_test_svc)
    
    res = res.append({'Preprocessing': preproc, 'Model': 'SVM', 'Precision': pres, 
                     'Recall': rec, 'F1-score': f1, 'Accuracy': acc}, ignore_index = True)
    
    return res

  

def training_rdg(X_train_rdg, X_test_rdg, y_train_rdg, y_test_rdg, nfolds , preproc):

    params = {
                     
                     'alpha' : [0.001, 0.01, 0.1, 1]
      }
    param_comb = 2
    rdg = RidgeClassifier()
    skf = StratifiedKFold(n_splits=nfolds, shuffle = True, random_state = 1001)
    random_search = RandomizedSearchCV(rdg, param_distributions=params, n_iter=param_comb, 
                                     n_jobs=-1, cv=skf.split(X_train_rdg,y_train_rdg), verbose=3, random_state=1001 )
        
    random_search.fit(X_train_rdg, y_train_rdg)
    best_param = random_search.best_params_
    
    print(best_param)
    svc = RidgeClassifier(alpha=best_param["alpha"]).fit(X_train_rdg, y_train_rdg)
    y_pred = svc.predict(X_test_rdg)
    
    res = pd.DataFrame(columns = ['Preprocessing', 'Model', 'Precision', 'Recall', 'F1-score', 'Accuracy'])

    f1 = f1_score(y_pred, y_test_rdg, average = 'weighted')
    pres = precision_score(y_pred, y_test_rdg, average = 'weighted')
    rec = recall_score(y_pred, y_test_rdg, average = 'weighted')
    acc = accuracy_score(y_pred, y_test_rdg)
    
    res = res.append({'Preprocessing': preproc, 'Model': 'Ridge', 'Precision': pres, 
                     'Recall': rec, 'F1-score': f1, 'Accuracy': acc}, ignore_index = True)
    
    return res
  
  
  

def training_xgb(X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb, nfolds , preproc):   
    # A parameter grid for XGBoost
    params = {           
                    'learning_rate': [0.01,0.02],
                    'n_estimators' : randint(low = 300, high = 1000),
                    'min_child_weight': [1, 5, 10],
                    'gamma': [0.5, 1, 1.5, 2, 5],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'max_depth': [3, 4, 5]
              }

    param_comb = 7
    xgb = XGBClassifier( objective='multi:softmax',
                     silent=True, nthread=1).fit(X_train_xgb, y_train_xgb)
    skf = StratifiedKFold(n_splits=nfolds, shuffle = True, random_state = 1001)
    random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, 
                                     n_jobs=-1, cv=skf.split(X_train_xgb,y_train_xgb), verbose=3, random_state=1001 )

    random_search.fit(X_train_xgb, y_train_xgb)
    best_param = random_search.best_params_
    
    print(best_param)
    
    xgb = XGBClassifier(learning_rate=best_param['learning_rate'], n_estimators=best_param['n_estimators'], 
                        objective='multi:softmax',
                         silent=True, nthread=1, min_child_weight=best_param['min_child_weight'],
                         gamma=best_param['gamma'] , subsample =best_param['subsample'],
                         colsample_bytree = best_param['colsample_bytree'] ,max_depth = best_param['max_depth']                        
                        ).fit(X_train_xgb, y_train_xgb)
    
    
    y_pred = xgb.predict(X_test_xgb)
    
    res = pd.DataFrame(columns = ['Preprocessing', 'Model', 'Precision', 'Recall', 'F1-score', 'Accuracy'])

    f1 = f1_score(y_pred, y_test_xgb, average = 'weighted')
    pres = precision_score(y_pred, y_test_xgb, average = 'weighted')
    rec = recall_score(y_pred, y_test_xgb, average = 'weighted')
    acc = accuracy_score(y_pred, y_test_xgb)
    
    res = res.append({'Preprocessing': preproc, 'Model': 'XGBoost', 'Precision': pres, 
                      'Recall': rec, 'F1-score': f1, 'Accuracy': acc}, ignore_index = True)
    
    return res


# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Evaluation

# COMMAND ----------

y = data['classes']
y_encoded = LabelEncoder().fit_transform(data['classes'])
print('Shape of label tensor:', y.shape)

# COMMAND ----------

set(data['classes'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline Model

# COMMAND ----------

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state = 42)

clf = LogisticRegressionCV(cv = folds, solver = 'saga',   max_iter=100,
                           multi_class = 'multinomial', n_jobs = -1, random_state = 42, penalty='l2')

df_res = pd.DataFrame(columns = ['Preprocessing', 'Precision', 'Recall', 'F1-score', 'Accuracy'])

# COMMAND ----------

feature_no=3500 # is selected based on F1-score of best performing model

# COMMAND ----------


# Bag of words approach
X = cntvectorizer(data)
ch2 = SelectKBest(chi2, k=feature_no)
X = ch2.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
df_res = df_res.append({'Preprocessing': 'Bag of words',
                       'Precision': precision,
                       'Recall': recall,
                       'F1-score': f1,
                       'Accuracy': accuracy}, ignore_index = True)

# TF_IDF approach. 1-gram
X = tfidf(data)
ch2 = SelectKBest(chi2, k=feature_no)
X = ch2.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
df_res = df_res.append({'Preprocessing': 'TF-IDF 1-gram',
                       'Precision': precision,
                       'Recall': recall,
                       'F1-score': f1,
                       'Accuracy': accuracy}, ignore_index = True)


# TF_IDF approach. 2-gram
X = tfidf(data, ngrams=2)
ch2 = SelectKBest(chi2, k=feature_no)
X = ch2.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
df_res = df_res.append({'Preprocessing': 'TF-IDF 2-gram',
                       'Precision': precision,
                       'Recall': recall,
                       'F1-score': f1,
                       'Accuracy': accuracy}, ignore_index = True)


# # Word2vec
# X = w2v(data)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
# df_res = df_res.append({'Preprocessing': 'Word2vec',
#                        'Precision': precision,
#                        'Recall': recall,
#                        'F1-score': f1,
#                        'Accuracy': accuracy}, ignore_index = True)

# COMMAND ----------

df_res

# COMMAND ----------

# MAGIC %md 
# MAGIC ###Finding Best Number of Features

# COMMAND ----------

# df_res = pd.DataFrame(columns = ['Preprocessing', 'Precision', 'Recall', 'F1-score', 'Accuracy','k'])

# for k in np.arange (2500,10000,500):
#     # TF_IDF approach. 1-gram
#     X = tfidf(data)
#     ch2 = SelectKBest(chi2, k=k)
#     X = ch2.fit_transform(X, y)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
#     df_res = df_res.append({'Preprocessing': 'TF-IDF 1-gram',
#                            'Precision': precision,
#                            'Recall': recall,
#                            'F1-score': f1,
#                            'Accuracy': accuracy,
#                            'k': k}, ignore_index = True)

# df_res

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Result

# COMMAND ----------

# # DataFrame for result evaluation
# folds = 5
# final_result = pd.DataFrame(columns = ['Preprocessing', 'Model', 'Precision', 'Recall', 'F1-score', 'Accuracy'])

# # Testing Count Vectorizer
# X = cntvectorizer(data)
# ch2 = SelectKBest(chi2, k=feature_no)
# X = ch2.fit_transform(X, y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# final_result = final_result.append(training_naive(X_train, X_test, y_train, y_test, 'Count Vectorize'), ignore_index = True)
# final_result = final_result.append(training_logreg(X_train, X_test, y_train, y_test, 'Count Vectorize'), ignore_index = True)
# final_result = final_result.append(training_svc(X_train, X_test, y_train, y_test, folds, 'Count Vectorize'), ignore_index = True)
# final_result = final_result.append(training_rdg(X_train, X_test, y_train, y_test, folds, 'Count Vectorize'), ignore_index = True)

# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=40)
# final_result = final_result.append(training_xgb(X_train, X_test, y_train, y_test, folds, 'Count Vectorize'), ignore_index = True)



# # Testing TF-IDF with 1-gram
# X = tfidf(data, ngrams = 1)
# ch2 = SelectKBest(chi2, k=feature_no)
# X = ch2.fit_transform(X, y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
# final_result = final_result.append(training_naive(X_train, X_test, y_train, y_test, 'TF-IDF 1-grams'), ignore_index = True)
# final_result = final_result.append(training_logreg(X_train, X_test, y_train, y_test, 'TF-IDF 1-grams'), ignore_index = True)
# final_result = final_result.append(training_svc(X_train, X_test, y_train, y_test, folds, 'TF-IDF 1-grams'), ignore_index = True)
# final_result = final_result.append(training_rdg(X_train, X_test, y_train, y_test, folds, 'TF-IDF 1-grams'), ignore_index = True)

# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=40)
# final_result = final_result.append(training_xgb(X_train, X_test, y_train, y_test, folds, 'TF-IDF 1-grams'), ignore_index = True)


# # Testing TF-IDF with 2-gram
# X = tfidf(data, ngrams = 2)
# ch2 = SelectKBest(chi2, k=feature_no)
# X = ch2.fit_transform(X, y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
# final_result = final_result.append(training_naive(X_train, X_test, y_train, y_test, 'TF-IDF 2-grams'), ignore_index = True)
# final_result = final_result.append(training_logreg(X_train, X_test, y_train, y_test, 'TF-IDF 2-grams'), ignore_index = True)
# final_result = final_result.append(training_svc(X_train, X_test, y_train, y_test, folds, 'TF-IDF 2-grams'), ignore_index = True)
# final_result = final_result.append(training_rdg(X_train, X_test, y_train, y_test, folds, 'TF-IDF 2-grams'), ignore_index = True)

# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=40)
# final_result = final_result.append(training_xgb(X_train, X_test, y_train, y_test, folds, 'TF-IDF 2-grams'), ignore_index = True)




# # Testing Word2vec
# X = w2v(data)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
# final_result = final_result.append(training_logreg(X_train, X_test, y_train, y_test, 'Word2vec'), ignore_index = True)
# final_result = final_result.append(training_svc(X_train, X_test, y_train, y_test, folds, 'Word2vec'), ignore_index = True)
# final_result = final_result.append(training_rdg(X_train, X_test, y_train, y_test, folds, 'Word2vec'), ignore_index = True)

# COMMAND ----------

#pd.DataFrame(final_result).sort_values(by=['F1-score','Accuracy'],axis=0, ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Result with Over-sampled Data

# COMMAND ----------

# # DataFrame for result evaluation
# folds = 5
# smote_result = pd.DataFrame(columns = ['Preprocessing', 'Model', 'Precision', 'Recall', 'F1-score', 'Accuracy'])



# # # Testing Count Vectorizer
# # X = cntvectorizer(data)
# # ch2 = SelectKBest(chi2, k=feature_no)
# # X = ch2.fit_transform(X, y)

# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# # # apply SMOTE for over sampling data
# # smote_over_sample = SMOTE(sampling_strategy='minority')
# # X_os = cntvectorizer(X_train)
# # X_os_test, y_os_test = smote_over_sample.fit_resample(X_os, y_test)


# # smote_result = smote_result.append(training_naive(X_os_test, y_os_test , y_train, y_test, 'Count Vectorize'), ignore_index = True)
# # smote_result = smote_result.append(training_logreg(X_os_test, y_os_test, y_train, y_test, 'Count Vectorize'), ignore_index = True)
# # smote_result = smote_result.append(training_svc(X_os_test, y_os_test, y_train, y_test, folds, 'Count Vectorize'), ignore_index = True)
# # smote_result = smote_result.append(training_rdg(X_os_test, y_os_test, y_train, y_test, folds, 'Count Vectorize'), ignore_index = True)

# # X_train, X_test, y_train, y_test = train_test_split(X_os_test, y_os_test_encoded, test_size=0.2, random_state=40)
# # smote_result = smote_result.append(training_xgb(X_train, X_test, y_train, y_test, folds, 'Count Vectorize'), ignore_index = True)



# # Testing TF-IDF with 1-gram
# X = tfidf(data, ngrams = 1)
# ch2 = SelectKBest(chi2, k=feature_no)
# X = ch2.fit_transform(X, y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# # apply SMOTE for over sampling data
# smote_over_sample = SMOTE(sampling_strategy='minority',random_state=2)
# X_os_train, y_os_train = smote_over_sample.fit_resample(X_train, y_train)

# smote_result = smote_result.append(training_naive(X_os_train, X_test, y_os_train, y_test, 'TF-IDF 1-grams'), ignore_index = True)
# smote_result = smote_result.append(training_logreg(X_os_train, X_test, y_os_train, y_test, 'TF-IDF 1-grams'), ignore_index = True)
# smote_result = smote_result.append(training_svc(X_os_train, X_test, y_os_train, y_test, folds, 'TF-IDF 1-grams'), ignore_index = True)
# smote_result = smote_result.append(training_rdg(X_os_train, X_test, y_os_train, y_test, folds, 'TF-IDF 1-grams'), ignore_index = True)


# X = tfidf(data, ngrams = 1)
# ch2 = SelectKBest(chi2, k=feature_no)
# X = ch2.fit_transform(X, y_encoded)
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=40)

# # apply SMOTE for over sampling data
# smote_over_sample = SMOTE(sampling_strategy='minority',random_state=2)
# X_os_train, y_os_train = smote_over_sample.fit_resample(X_train, y_train)

# final_result = final_result.append(training_xgb(X_os_train, X_test, y_os_train, y_test, 'TF-IDF 1-grams'), ignore_index = True)


# # Testing TF-IDF with 2-gram
# X = tfidf(data, ngrams = 2)
# ch2 = SelectKBest(chi2, k=feature_no)
# X = ch2.fit_transform(X, y)
# X_train, X_test, y_train, y_test = train_test_split(X_os_test, y_os_test, test_size=0.2, random_state=40)
# smote_result = smote_result.append(training_naive(X_train, X_test, y_train, y_test, 'TF-IDF 2-grams'), ignore_index = True)
# smote_result = smote_result.append(training_logreg(X_train, X_test, y_train, y_test, 'TF-IDF 2-grams'), ignore_index = True)
# smote_result = smote_result.append(training_svc(X_train, X_test, y_train, y_test, folds, 'TF-IDF 2-grams'), ignore_index = True)
# smote_result = smote_result.append(training_rdg(X_train, X_test, y_train, y_test, folds, 'TF-IDF 2-grams'), ignore_index = True)

# X_train, X_test, y_train, y_test = train_test_split(X_os_test, y_os_test_encoded, test_size=0.2, random_state=40)
# smote_result = smote_result.append(training_xgb(X_train, X_test, y_train, y_test, folds, 'TF-IDF 2-grams'), ignore_index = True)




# # Testing Word2vec
# X = w2v(data)
# X_train, X_test, y_train, y_test = train_test_split(X_os_test, y_os_test, test_size=0.2, random_state=40)
# smote_result = smote_result.append(training_logreg(X_train, X_test, y_train, y_test, 'Word2vec'), ignore_index = True)
# smote_result = smote_result.append(training_svc(X_train, X_test, y_train, y_test, folds, 'Word2vec'), ignore_index = True)
# smote_result = smote_result.append(training_rdg(X_train, X_test, y_train, y_test, folds, 'Word2vec'), ignore_index = True)



# COMMAND ----------

#pd.DataFrame(smote_result).sort_values(by=['F1-score','Accuracy'],axis=0, ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Best Model

# COMMAND ----------

# Testing TF-IDF with 1-gram
X = tfidf(data, ngrams = 1)
ch2 = SelectKBest(chi2, k=3000)
X = ch2.fit_transform(X, y)
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y, test_size=0.2, random_state=40)

folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)   
clf = LogisticRegressionCV(cv = folds, solver = 'saga', multi_class = 'multinomial', n_jobs = -1, penalty='l2')   
clf.fit(X_train_log, y_train_log)
res = pd.DataFrame(columns = ['Preprocessing', 'Model', 'Precision', 'Recall', 'F1-score', 'Accuracy'])
y_pred = clf.predict(X_test_log)
f1 = f1_score(y_pred, y_test_log, average = 'weighted')
pres = precision_score(y_pred, y_test_log, average = 'weighted')
rec = recall_score(y_pred, y_test_log, average = 'weighted')
acc = accuracy_score(y_pred, y_test_log)
res = res.append({'Preprocessing': 'TF-IDF 1-grams', 'Model': f'Logistic Regression', 'Precision': pres, 
                     'Recall': rec, 'F1-score': f1, 'Accuracy': acc}, ignore_index = True)


# COMMAND ----------

res

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Prediction

# COMMAND ----------

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from time import time

# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer(max_df=0.5,binary=False,max_features=3000,ngram_range=(1,1))),
    ('tfidf', TfidfTransformer(norm='l2',use_idf=True)),
    ('clf', LogisticRegression(multi_class = 'multinomial',solver='saga',penalty='l2')),
])
#hard coding best parameters to decrease testing runtime
parameters = {}


# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
# parameters = {
#     'vect__max_df': (0.5, 0.75),
#     'vect__max_features': (None,3000,5000,6000,10000),
#     'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
#     'vect__binary':[True,False],
#     'tfidf__use_idf': (True,False),
#     #'tfidf__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
#     'tfidf__norm': ( 'l1','l2'),
#     #'clf__max_iter': (20,),
#     #'clf__alpha': (0.00001, 0.000001),
#     #'clf__penalty': ('l2')    
#     # 'clf__max_iter': (10, 50, 80),
# }


grid_search = GridSearchCV(pipeline, parameters, cv=5,n_jobs=-1, verbose=1)


print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
t0 = time()
data['stmlemtext']=data['stmlemtext'].astype(str)

print(parameters)
grid_search.fit(data['stmlemtext'], y)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# COMMAND ----------

df_holdout['txtprocessed'] = df_holdout['text'].apply(clean_text)
df_holdout['txtprocessed'] = df_holdout['txtprocessed'].str.replace('\d+', '')
df_holdout['txtprocessed'] = df_holdout['txtprocessed'].apply(wordnet_lemmatize)


y_h = df_holdout['classes']
y_pred_h = grid_search.predict(df_holdout['txtprocessed'])

# COMMAND ----------

y_h

# COMMAND ----------

res = pd.DataFrame(columns = ['Preprocessing', 'Model', 'Precision', 'Recall', 'F1-score', 'Accuracy'])

f1 = f1_score(y_h, y_pred_h, average = 'weighted')
pres = precision_score( y_h, y_pred_h, average = 'weighted')
rec = recall_score( y_h,y_pred_h, average = 'weighted')
acc = accuracy_score( y_h,y_pred_h)
res = res.append({'Preprocessing': 'TF-IDF vec 1-grams', 'Model': f'Logistic Regression', 'Precision': pres, 
                     'Recall': rec, 'F1-score': f1, 'Accuracy': acc}, ignore_index = True)

# COMMAND ----------

res

# COMMAND ----------

encoded_y_test_h =  LabelEncoder().fit_transform(y_h)
encoded_y_pred_h =  LabelEncoder().fit_transform(y_pred_h)

confusion  = confusion_matrix(encoded_y_test_h, encoded_y_pred_h)  # the true class is the first argument in all these functions
fig, ax = plt.subplots(figsize=(15,10))
plot_confusion_matrix(confusion, classes=df_columns, normalize=True)
#plt.title(labels)
display()

# COMMAND ----------

prediction_df = pd.DataFrame(columns=['text','prediction'])
prediction_df['text'] = df_holdout['txtprocessed']
prediction_df['label'] = pd.DataFrame(y_h)
prediction_df['prediction'] = pd.DataFrame(y_pred)

# COMMAND ----------

fig, ax =plt.subplots(1,2)
plt.rcParams['figure.figsize'] = (15, 30)
g=sns.barplot(data=pd.DataFrame((pd.get_dummies(prediction_df['label']).iloc[:,0:].sum(axis=0))).sort_values(by=0,ascending=False, inplace=False).transpose(),ax=ax[0])
w=sns.barplot(data=pd.DataFrame((pd.get_dummies(prediction_df['prediction']).iloc[:,0:].sum(axis=0))).sort_values(by=0,ascending=False, inplace=False).transpose(),ax=ax[1])

a=g.set_xticklabels(g.get_xticklabels(),rotation=35)
display()

# COMMAND ----------

prediction_df.head(10)

# COMMAND ----------

