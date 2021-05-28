# -*- coding: utf-8 -*-
"""
Created on Mon May  3 16:54:55 2021

@author: Komali Srinivas
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import re

from nltk.corpus import stopwords
import nltk
from string import punctuation
from nltk.stem.porter import *
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


stop_words = list(set(stopwords.words('english')))+list(punctuation)+['``', "'s", "...", "n't"]

os.getcwd()
os.chdir(r'C:\Users\Komali Srinivas\Day1\hacky2\data')

#columns = ['target','ids','date','flag','user','text']

df = pd.read_csv('train.csv')


#data = pd.read_csv('train.csv')
#data.columns = columns

#data = data.sample(n=5000,random_state=22)

type(df.tweet)

def remove_pattern(input_txt,pattern):
    r = re.findall(pattern, input_txt)
    for match in r:
        input_txt = re.sub(match, '', input_txt)
    return input_txt

df['clean_tweet']= df['tweet'].apply(lambda row:remove_pattern(row, "@[\w]*"))

#data['tokenized_text'] = nltk.word_tokenize(data['clean_text'][1013875])

df['tokenized_text'] = 0

#for i in data.index:
#    data['tokenized_text'][i] = nltk.word_tokenize(data['clean_text'][i])

df['tokenized_text']= df.apply(lambda data: nltk.word_tokenize(data['clean_text']), axis=1)

# stopword removal
df['tokenized_text']=df.apply(lambda data:[word for word in data['tokenized_text'] if not word in stop_words],axis=1)


# stemming words
stemmer = PorterStemmer()

df['tokenized_text'] = df['tokenized_text'].apply(lambda x: [stemmer.stem(i) for i in x])

df['tokenized_text'] = df['tokenized_text'].apply(lambda x: ' '.join(x))

print(df['tokenized_text'].head())

df.shape

all_words = df['tokenized_text'].to_string()
wordcloud = 0
# generate wordcloud object

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
# plot wordcloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

neg_words = ' '.join([text for text in data['tokenized_text'][data['target'] == 0]])

neg_wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(neg_words)

plt.imshow(neg_wordcloud, interpolation='bilinear')
plt.axis('off')

ratio = int(len(data)*0.75)

# logistic regression model
logreg = LogisticRegression(random_state=2)

# Code starts here

# TF-IDF feature matrix
tfidf_vectorizer = TfidfVectorizer(max_df=0.90,min_df=2, max_features=1000, stop_words='english')

# fit and transform tweets
tweets = tfidf_vectorizer.fit_transform(data['tokenized_text'])
                
data['target'][data['target']==4] = 1

X_train= tweets[:ratio,:]
X_test = tweets[ratio:,:]
y_train = data['target'].iloc[:ratio]
y_test = data['target'].iloc[ratio:]

logreg = LogisticRegression()

logreg.fit(X_train,y_train)

prediction = logreg.predict_proba(X_test)

prediction_int = prediction[:,1] >= 0.3

prediction_int = prediction_int.astype(np.int)

accuracy = accuracy_score(y_test, prediction_int)

tb_polarity = []

for sentence in data['tokenized_text']:
    temp = TextBlob(sentence)
    tb_polarity.append(temp.sentiment[0])

data['tb_polarity'] = tb_polarity

len(tb_polarity) == len(data)

round(tb_polarity[10], 2) == -0.7

analyser = SentimentIntensityAnalyzer()

vs_polarity = []

for sentence in data['tokenized_text']:
    vs_polarity.append(analyser.polarity_scores(sentence)['compound'])

data['vs_polarity'] = vs_polarity

len(vs_polarity) == len(data)

round(data.iloc[0,-1],2) == 0.46





        