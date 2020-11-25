import vectorization
import sklearn
from sklearn.feature_extraction.text import CountVectorizer;
from sklearn.feature_extraction.text import TfidfVectorizer;
import preprocessing
import pandas as pd

print("Glue on a roach baby!!!!!")

def binarizeRating(rating):
    if rating >=3.5:
        return 1;
    return 0;

all_df = preprocessing.getDfFromJSON('All_Beauty.json.gz');
sub_df = all_df.iloc[:800,];

words_df = preprocessing.filterOutNonString(sub_df,"reviewText");
print(len(words_df['reviewText']));
words_df['documents']=words_df['reviewText'].map(preprocessing.preprocess);

words_df;

all_words =words_df['documents']
all_words = [word for words in all_words for word in words ]
doc_list = words_df['documents']

docList = [" ".join(doc) for doc in words_df['documents']];
words_df['documentStrings'] = docList;
v = CountVectorizer(stop_words='english');
v.fit(all_words);
sparceVector = v.transform(docList)
print(sparceVector)
# words_df["sparceVector"] = sparceVector.toarray();

#now to make a column of data to classify agains
words_df["pos_neg"] = words_df['overall'].map(binarizeRating)
words_df["sparceVector"]
import numpy as np
xTrain, xTest,yTrain, yTest = sklearn.model_selection.train_test_split(words_df['sparceVector'], words_df['pos_neg'],test_size = .2, train_size = .1);
xTrain[43]

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(xTrain, yTrain)
