import spacy
import sys
import nltk
from nltk.tokenize import word_tokenize
import gzip
import json
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer

stopwords = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']);

def filterOutNonString(df, column):
    words_only = df[column];
    print(len(words_only))
    bin = [type(words)==str for words in words_only]
    not_words = [i for i, x in enumerate(bin) if x == False]
    for indx in not_words:
        df = df.drop([indx])
    return df

def getDfFromJSON(path):
    data = []
    with gzip.open(path) as f:
        for l in f:
            data.append(json.loads(l.strip()))
    df = pd.DataFrame.from_dict(data);
    df = df[['overall', 'reviewText','summary']];
    return df

def tokenize(review):
    # tokenization
    words =word_tokenize(review);
    #remove digits and other symbols except "@"--used to remove email
    words = [re.sub(r"[^A-Za-z@]", "", word) for word in words]
    #e websites and email address
    words = [re.sub(r'\S+com', '', word) for word in words]
    words = [re.sub(r'\S+@\S+', '', word) for word in words]
    #remove empty spaces
    words = [word for word in words if word!='']
    return words;

def lowerCaseList(words_list):
    return [word.lower() for word in words_list];

def removeStopWords(text_list, stop_list):
    return [word for word in text_list if word not in stop_list]

def lemmatizeList(words_list,lemmatizer):
    return [lemmatizer(word) for word in words_list]

def lemmatizer(word):
    return nlp(word)[0].lemma_

def preprocessForSentimentAnalsis(review,stop_words,lemmatizer):
    print("    Input String       : ",review)
    words = tokenize(review);
    stop_words  = lowerCaseList(stop_words);
    input_words = lowerCaseList(words)
    print("    Tokenized string   : ",input_words)
    stoplessWords = removeStopWords(input_words, stop_words);
    print("    Stopless Words     : ",stoplessWords)
    lemmatizedWords = lemmatizeList(stoplessWords, lemmatizer)
    print("    Lemmatized Words   : ",lemmatizedWords)
    return lemmatizedWords

def preprocessDocumentList(document_array,stop_words,lemmatizer):
    return [preprocessForSentimentAnalsys(document, stopwords, lemmatizer) for document in document_list]

def preprocess(review):
    try:
        return preprocessForSentimentAnalsis(review, stopwords, lemmatizer);
    except(TypeError):
        print(type(review))
        return False


# print("Cleaning Negative String...");
# preprocess("This product had a very rough start!");
# print("Cleaning Positive String...");
# preprocess("I immediately knew I was going to enjoy this!");

#                                 Feature            Labels
# allWords        = ['w1', 'w2' , 'w3',  ... ,'wn'] [ 5  ]
# review1 :         [0,     0   ,  0  ,  ... ,  1 ] [ 2  ]
# review2 :         [1,     0   ,  1  ,  ... ,  0 ] [ 4  ]
#                                 :
#                                 :
# reviewn:          [0,     0   ,  0  , ... ,  1 ] [ 4  ]
# review2 :         [1,     0   ,   1,  ... ,  0 ] [  1 ]
