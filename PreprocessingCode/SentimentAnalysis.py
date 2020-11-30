import preprocessing
import vectorization
import pandas as pd;


# words_df = preprocessing.getDfFromJSON('All_Beauty.json.gz')
# words_df = words_df[:10] #just testing on a small substring of data
words_df = pd.read_csv('All_Beauty1.csv')
#at this point words df is just a column of review texts and their associated scores
# preprocessing.preprocessForSentimentAnalsis(words_df['reviewText'][4], preprocessing.stopwords,preprocessing.lemmatizer);
# words_df['documents']=words_df['reviewText'].map(preprocessing.preprocess)
# words_df = words_df[words_df['documents'] != False]
#documents column now is just the preprocessed words stripped of fluff, ready to be turned into a sparse matrix
#First we just need a list of all of the words

#now we construct our CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer;
from sklearn.feature_extraction.text import TfidfVectorizer;

#generate a sparce array from the things
words_df['documents'] = [" ".join(preprocessing.tokenize(doc)).split(" ") for doc in words_df['documents']]
all_words = vectorization.getAllWordsFromDF(words_df, 'documents')
docList= [" ".join(doc) for doc in words_df['documents']]

# docList = vectorization.ListToString(words_df,'documents')
v,sparceVector = vectorization.vectorize(CountVectorizer, all_words, docList)
sv_array = sparceVector.toarray()

#now we just need to form our labels in whatever way we want them to
words_df["pos_neg"] = words_df['overall'].map(vectorization.binarizeRating)
import sklearn
import numpy as np
xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(sv_array,list(words_df['pos_neg']),test_size = .3);

ytrain = np.array(yTrain)
ytest = np.array(yTest)
ytrain.shape
xTrain.shape
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import normalize

# ytrain=ytrain/5.0
# ytest = ytest/5.0

modelAmazon = Sequential()
modelAmazon.add(Dense(50, activation = "relu", input_shape=(1749, )))
# Hidden - Layers
modelAmazon.add(Dropout(0.3, noise_shape=None, seed=None))
modelAmazon.add(Dense(50, activation = "relu"))
modelAmazon.add(Dropout(0.3, noise_shape=None, seed=None))
modelAmazon.add(Dense(50, activation = "relu"))
# Output- Layer
modelAmazon.add(Dense(1, activation = "sigmoid"))
modelAmazon.summary()

modelAmazon.compile(
 optimizer = "adam",
 loss = "binary_crossentropy",
 metrics = ["accuracy"]
)

results = modelAmazon.fit(
 xTrain, ytrain,
 epochs= 10,
 batch_size = 700,
 validation_data = (xTest, ytest)
)

print(np.mean(results.history["accuracy"]))

from keras.datasets import imdb
index = imdb.get_word_index()
values = index.values()
min(values)
max(values)


(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words = 10000)
len(training_data[6])

len(training_data[6])
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)
data

def vectorize(sequences, dimension = 10000):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1
  return results

data = vectorize(data)
training_targets[8]
targets = np.array(targets).astype("float32")

test_x = data[:10000]
type(test_x[0][0])
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

print(test_y)
print(type(test_y))
print(type(test_y[0]))
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential()
model.add(Dense(50, activation = "relu", input_shape=(10000, )))
# Hidden - Layers
model.add(Dropout(0.3, noise_shape=None, seed=None))
model.add(Dense(50, activation = "relu"))
model.add(Dropout(0.3, noise_shape=None, seed=None))
model.add(Dense(50, activation = "relu"))
# Output- Layer
model.add(Dense(1, activation = "sigmoid"))
model.summary()

model.compile(
 optimizer = "adam",
 loss = "binary_crossentropy",
 metrics = ["accuracy"]
)

results = model.fit(
 train_x, train_y,
 epochs= 5,
 batch_size = 500,
 validation_data = (test_x, test_y)
)

print(np.mean(results.history["accuracy"]))
