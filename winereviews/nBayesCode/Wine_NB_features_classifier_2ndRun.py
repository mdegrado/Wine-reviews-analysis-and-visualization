import pandas as pd
# from pandas import options
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from matplotlib.style import use
use('ggplot')


df = pd.read_csv('C:/data/wine/wine-reviews/winemag-data-130k-v2.csv')
df.head(5)
df = df.dropna(subset=['price', 'points'])
df[['price', 'points']].hist(layout=(2,1))
plt.show()
plt.scatter(df['points'], df['price'])
plt.xlabel('Points')
plt.ylabel('Price')
plt.show()
df = df.dropna(subset=['description'])  # drop all NaNs

df_sorted = df.sort_values(by='points', ascending=True)  # sort by points

num_of_wines = df_sorted.shape[0]  # number of wines
worst = df_sorted.head(int(0.30*num_of_wines))  # 30 % worst by pts
best = df_sorted.tail(int(0.30*num_of_wines))  # 30 % best by pts
plt.hist(df['points'], color='grey', label='All')
plt.hist(worst['points'], color='red', label='Worst')
plt.hist(best['points'], color='green', label='Best')
plt.legend()
plt.show()
from nltk.tokenize import word_tokenize
from nltk import FreqDist, NaiveBayesClassifier
from random import shuffle
worst['words'] = worst['description'].apply(func=lambda text: word_tokenize(text.lower()))
best['words'] = best['description'].apply(func=lambda text: word_tokenize(text.lower()))
worst = worst.dropna(subset=['words'])  # drop all NaNs
best = best.dropna(subset=['words'])  # drop all NaNs
#nltk.download('punkt')
import nltk
#nltk.download('popular')
all_words = []  # initialize list of all words
# add all words from 'worst' dataset
for description in worst['words'].values:
    for word in description:
        all_words.append(word)
# add all words from 'best' dataset
for description in best['words'].values:
    for word in description:
        all_words.append(word)
all_words = FreqDist(all_words)  # make FreqList
words_features = list(all_words.keys())[:4000]  # select 4000 most frequent words as words features
def find_features(doc):
    """Function for making features out of the text"""
    words = set(doc)  # set of words in description
    features = {}  # feature dictionary
    for w in words_features:  # check if any feature word is presented
        features[w] = bool(w in words)  # write to feature vector
    return features  # return feature vector
featureset = ([(find_features(description), 'worst') for description in worst['words']] +
              [(find_features(description), 'best') for description in best['words']])
shuffle(featureset)  # randomly shuffle dataset
classifier = NaiveBayesClassifier.train(labeled_featuresets=featureset)
classifier.show_most_informative_features(100)