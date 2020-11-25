import preprocessing

words_df = preprocessing.getDfFromJSON('All_Beauty.json.gz')
words_df = words_df[:10]

def testFunc(l):
    print(type(l))
    return 1;

words_df.keys();
preprocessing.preprocessForSentimentAnalsis(words_df['reviewText'][4], preprocessing.stopwords,preprocessing.lemmatizer);
words_df['documents']=words_df['reviewText'].map(preprocessing.preprocess)
words_df;

import vectorization
from sklearn.feature_extraction.text import CountVectorizer;
from sklearn.feature_extraction.text import TfidfVectorizer;

def vectorizer(document):
    return vectorization.vectorize(document, TfidfVectorizer)

import pandas as pd
all_words =words_df['documents']
all_words = [word for words in all_words for word in words ]

frequencyVector = vectorizer(all_words)
frequencyVector.get_feature_names()
