import preprocessing
import vectorization

print(vectorization)
words_df = preprocessing.getDfFromJSON('All_Beauty.json.gz')
words_df = words_df[:10] #just testing on a small substring of data

#at this point words df is just a column of review texts and their associated scores
preprocessing.preprocessForSentimentAnalsis(words_df['reviewText'][4], preprocessing.stopwords,preprocessing.lemmatizer);
words_df['documents']=words_df['reviewText'].map(preprocessing.preprocess)

#documents column now is just the preprocessed words stripped of fluff, ready to be turned into a sparse matrix
#First we just need a list of all of the words

#now we construct our CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer;
from sklearn.feature_extraction.text import TfidfVectorizer;

all_words = vectorization.getAllWordsFromDF(words_df, 'documents')
docList = vectorization.ListToString(words_df,'documents')
v,sparceVector = vectorization.vectorize(CountVectorizer, all_words, docList)
sv_array = sparceVector.toarray()

#now we just need to form our labels in whatever way we want them to
words_df["pos_neg"] = words_df['overall'].map(vectorization.binarizeRating)
import sklearn
import numpy as np
xTrain, xTest,yTrain, yTest = sklearn.model_selection.train_test_split(sv_array,list(words_df['overall']),test_size = .3);

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(xTrain, yTrain)
