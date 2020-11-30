import preprocessing
import sys

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

try:
    filename = str(sys.argv[1])
    noOfTerms = int(sys.argv[2])
    assert(type(noOfTerms)==int)
    print("Taking ",noOfTerms," from amazon dataset ",filename )
    words_df = preprocessing.getDfFromJSON(filename)
    if noOfTerms!=1:
        words_df = words_df[:noOfTerms]
    else:
        noOfTerms = len(words_df)
    # for text in words_df['reviewText'][4]
    # preprocessing.preprocessForSentimentAnalsis(words_df['reviewText'][4], preprocessing.stopwords,preprocessing.lemmatizer);
    words_df['documents']=words_df['reviewText'].map(preprocessing.preprocess)
    words_df.to_csv(filename[:-8]+str(noOfTerms)+'.csv',index=False)
except(KeyError, IndexError):
    print("No argument given, or file not found")
