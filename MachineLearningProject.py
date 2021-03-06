# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZRGqn9WaUBKr7UViGqwuToJhNK09Dc9q
"""

# Commented out IPython magic to ensure Python compatibility.
import os
import json
import gzip
import pandas as pd
from urllib.request import urlopen
import re
import wget
import nltk
from nltk.tokenize import word_tokenize
from nltk import punkt

import matplotlib.pyplot as plt
# %matplotlib inline

import random
import numpy as np
from tqdm import tqdm_notebook as tqdm
from collections import defaultdict

#all_beauty = wget.download('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/All_Beauty.json.gz')

data = []
with gzip.open('All_Beauty.json.gz') as f:
    for l in f:
        data.append(json.loads(l.strip()))

# total length of list, this number equals total number of products
print(len(data))

# first row of the list
print(data[0])

# convert list into pandas dataframe

df = pd.DataFrame.from_dict(data)

print(len(df))

df.head()
# Selecting the "survived" column
review_data = df["reviewText"].copy()

review_data.describe()

#for i in review_data:
  #print(i, "--")

features = df.iloc[:,6].values
features = features[:15000]
labels = df.iloc[:, 0].values
labels = labels[:15000]

#print(features[7])
print(labels[7])

processed_features = []

#This section is made to clean the data we have
for review in features:

    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(review))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    # tokenization
    processed_feature = word_tokenize(processed_feature)

    processed_features.append(processed_feature)

#This section is made to normalize the data we've collected and cleaned
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
#from nltk.tag. import averaged_perceptron_tagger
from sklearn.feature_extraction.text import TfidfVectorizer

#In the future, will make our own stop_words list to improve accuracy
#stop_words=['in','of','at','a','the', 'if', 'is', 'it']
def removeStopWords(text_list, stop_list):
    return [word for word in text_list if word not in stop_list]

#def lowerCaseList(words_list):
    #return [word.lower() for word in words_list];

stopwords = nltk.corpus.stopwords.words('english')
#stop_words  = lowerCaseList(stopwords)
#input_words = lowerCaseList(processed_features)
processed_features = removeStopWords(processed_features, stopwords)


#vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
#processed_features = vectorizer.fit_transform(processed_features[:5000]).toarray()


#Normalization
def get_lemmatized_text(corpus):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review]) for review in corpus]

processed_features = get_lemmatized_text(processed_features)


#Using ngrams to do additional context association for the data, including negation words.
from sklearn.feature_extraction.text import CountVectorizer

ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(processed_features)
#train feautures
X = ngram_vectorizer.transform(processed_features[:10000])
#test features
X_test = ngram_vectorizer.transform(processed_features[10000:])


from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, labels[:3000], test_size=0.25, random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X_train, X_val, y_train, y_val = train_test_split(
    X, labels[:10000], train_size=0.75
)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print("Accuracy for C=%s: %s"
          % (c, accuracy_score(y_val, lr.predict(X_val))))

# Accuracy for C=0.01: 0.88416
# Accuracy for C=0.05: 0.892
# Accuracy for C=0.25: 0.89424
# Accuracy for C=0.5: 0.89456
# Accuracy for C=1: 0.8944

final_ngram = LogisticRegression(C=0.5)
final_ngram.fit(X, labels[:10000])
print("Final Accuracy: %s"
      % accuracy_score(labels[10000:], final_ngram.predict(X_test)))

predictions = final_ngram.predict(X_val)

print(predictions, np.size(predictions))

# Final Accuracy: 0.898

#######################################################################################




##################################################################
from sklearn.neighbors import KNeighborsClassifier

#text_classifier = KNeighborsClassifier(n_neighbors=6)
#text_classifier.fit(X_train, y_train)

#predictions = text_classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#print(predictions)

#print(confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))
#print(accuracy_score(y_test, predictions))




from sklearn.feature_extraction.text import CountVectorizer
#
cv = CountVectorizer(binary=True)
cv.fit(processed_features)
X = cv.transform(processed_features[:10000])
X_test = cv.transform(processed_features[10000:])
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
#
X_train, X_val, y_train, y_val = train_test_split(
     X, labels[:10000], train_size=0.75
 )
#
for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print("Accuracy for C=%s: %s"
           % (c, accuracy_score(y_val, lr.predict(X_val))))


final_model = LogisticRegression(C=0.05)
final_model.fit(X, labels[:10000])
print ("Final Accuracy: %s"
        % accuracy_score(labels[10000:], final_model.predict(X_test)))

feature_to_coef = {
     word: coef for word, coef in zip(
        cv.get_feature_names(), final_model.coef_[0]
     )
 }
for best_positive in sorted(
        feature_to_coef.items(),
        key=lambda x: x[1],
        reverse=True)[:20]:
     print(best_positive)

for best_negative in sorted(
        feature_to_coef.items(),
        key=lambda x: x[1])[:20]:
    print(best_negative)


print ("Final Accuracy: %s",
        final_model.predict(X_test))
# print(X_test[0])
# #for key, value in feature_to_coef.items():
#   #print("key: ", key, " value: ", value)
# #print(feature_to_coef.items())
#
print(feature_to_coef.get('amazing'))

maxVal = feature_to_coef.get(max(feature_to_coef, key=feature_to_coef.get))
minVal = feature_to_coef.get(min(feature_to_coef, key=feature_to_coef.get))
difference = maxVal-minVal
# #print((maxVal+abs(minVal))/difference)
#
def normalize(valRange, minVal, positivity):
  return (positivity+abs(minVal))/valRange
#
#
# #print(normalize(difference, minVal, minVal))
#wordPositivity = {}
# for key, value in feature_to_coef.items():
#   #print(key, value)
#   wordPositivity[key] = normalize(difference, minVal, value)
# print(wordPositivity['perfect'])
#
# print(wordPositivity)