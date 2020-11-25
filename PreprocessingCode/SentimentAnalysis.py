import preprocessing
words_df = preprocessing.getDfFromJSON('All_Beauty.json.gz')
words_df = words_df[:10] #just testing on a small substring of data

#at this point words df is just a column of review texts and their associated scores
preprocessing.preprocessForSentimentAnalsis(words_df['reviewText'][4], preprocessing.stopwords,preprocessing.lemmatizer);
words_df['documents']=words_df['reviewText'].map(preprocessing.preprocess)

#documents column now is just the preprocessed words stripped of fluff, ready to be turned into a sparse matrix

#First we just need a list of all of the words
def getAllWordsFromDF(df, col):
    return [word for words in words_df['documents'] for word in words]

def ListToString(df, cpl):
    return [" ".join(doc) for doc in words_df['documents']];

def vectorize(vectorizer, vocabulary, list):
    v =vectorizer(stop_words='english')
    v.fit(vocabulary)
    sparse_vector = v.transform(list)
    return v, sparse_vector

#now we construct our CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer;
from sklearn.feature_extraction.text import TfidfVectorizer;

all_words = getAllWordsFromDF(words_df, 'documents')
docList = ListToString(words_df,'documents')
print(len(docList))
v,sparceVector = vectorize(CountVectorizer, all_words, docList)
sv_array = sparceVector.toarray()


#now we just need to form our labels in whatever way we want them to
def binarizeRating(rating):
    if rating >=3.5:
        return 1;
    return 0;

words_df["pos_neg"] = words_df['overall'].map(binarizeRating)
import sklearn
import numpy as np
xTrain, xTest,yTrain, yTest = sklearn.model_selection.train_test_split(sv_array,list(words_df['overall']),test_size = .3);

print(list(words_df['pos_neg']))
print(yTrain)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(xTrain, yTrain)


# v = CountVectorizer(stop_words='english');
# v.fit(all_words);
# sparceVector = v.transform(docList)
# print(sparceVector.toarray())

#now to vectorize these words


def vectorizer(document):
    return vectorization.vectorize(document, TfidfVectorizer)

import pandas as pd
all_words =words_df['documents']
all_words = [word for words in all_words for word in words ]
frequencyVector = vectorizer(all_words)
frequencyVector.get_feature_names()
