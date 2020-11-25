import sklearn
from sklearn.feature_extraction.text import CountVectorizer;
from sklearn.feature_extraction.text import TfidfVectorizer;
import pandas as pd

#this file contains all the functions needed to construct meaningful data from preprocessed words


all_df = preprocessing.getDfFromJSON('All_Beauty.json.gz');
sub_df = all_df.iloc[:800,];

words_df = preprocessing.filterOutNonString(sub_df,"reviewText");
print(len(words_df['reviewText']));
words_df['documents']=words_df['reviewText'].map(preprocessing.preprocess);


all_words =words_df['documents']
all_words = [word for words in all_words for word in words ]
doc_list = words_df['documents']

docList = [" ".join(doc) for doc in words_df['documents']];
words_df['documentStrings'] = docList;
v = CountVectorizer(stop_words='english');
v.fit(all_words);
sparceVector = v.transform(docList)
print(sparceVector)
words_df["sparceVector"] = sparceVector.toarray();

#now to make a column of data to classify agains
words_df["pos_neg"] = words_df['overall'].map(binarizeRating)
words_df["sparceVector"]


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(xTrain, yTrain)
