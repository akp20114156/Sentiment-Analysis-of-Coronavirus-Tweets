import numpy as np 
import pandas as pd 
import nltk
from collections import Counter , defaultdict
import requests
import re
import time
import matplotlib.pyplot as plt
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

start_time = time.time()
stopwords = requests.get("https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt").content.decode('utf-8').split("\n")

#Q1.Convert the messages to lower case, replace non-alphabetical characters with whitespaces ensure that the words of a message are separated by a single whitespace.

tweets_df = pd.read_csv("../data/text_data/Corona_NLP_train.csv",encoding='latin-1')

print('Possible sentiments',tweets_df.Sentiment.unique())
print(tweets_df.Sentiment.value_counts())
s = tweets_df.groupby(['TweetAt','Sentiment'])['Sentiment'].count().reset_index(name='count').sort_values(ascending=False,by='count')
print(s.loc[s['Sentiment']=='Extremely Positive'].head(1))

#Convert the messages to lower case
tweets_df.OriginalTweet = tweets_df.OriginalTweet.str.lower()
#replace non-alphabetical characters with whitespaces
tweets_df.OriginalTweet = tweets_df.OriginalTweet.replace('[^a-zA-Z]', ' ', regex=True)
#words of a message are separated by a single whitespace.
tweets_df.OriginalTweet = tweets_df.OriginalTweet.str.replace(' +', ' ')

# Q2 Tokenize the tweets (i.e. convert each into a list of words)
tweets_df['TokenisedTweet']= tweets_df['OriginalTweet'].apply(lambda row: str(row).split())

#count the total number of all words (including repetitions)
print('Number of words including repetitions',tweets_df.TokenisedTweet.str.len().sum())

#number of all distinct words
results = set()
tweets_df.TokenisedTweet.apply(results.update)
print('Number of distinct words',len(results))

#10 most frequent words
word_dict = {}
for i in tweets_df['TokenisedTweet']:
    for j in i:
        if j not in word_dict:
            word_dict[j] = 1
        else:
            word_dict[j] += 1
print('Ten most frequent words',Counter(word_dict).most_common(10))

#Remove stop words, words with 2 characters and recalculate the number of all words (including repetitions) and the 
#10 most frequent words in the modified corpus

# processing on stop words similiar to that of tweets
stop_df = pd.DataFrame(stopwords)
stop_df[0]= stop_df[0].str.lower()
stop_df[0] = stop_df[0].replace('[^a-zA-Z]', ' ', regex=True)
stop_df[0] = stop_df[0].str.replace(' +', ' ')

# tokenise tweets after removing stop words
stoplist = set(list(stop_df[0]))

tweets_df['TokenisedTweetWordRemoval'] = tweets_df['TokenisedTweet'].apply(lambda row: [item for item in row if item not in stoplist])
tweets_df['TokenisedTweetWordRemoval'] = tweets_df['TokenisedTweetWordRemoval'].apply(lambda row: [item for item in row if len(item)>2])

print('Number of words after preprocessing',tweets_df.TokenisedTweetWordRemoval.str.len().sum())
results = set()
tweets_df.TokenisedTweetWordRemoval.apply(results.update)
print('Number of distinct words after preprocessing',len(results))

#10 most frequent words after stop word removal 
# Most frequent word solved via - for loop and vectorisation both methods. However, for loop method is faster 
start_time = time.time()
word_dict = {}

for i in tweets_df['TokenisedTweetWordRemoval']:
    for j in i:
        if j not in word_dict:
            word_dict[j] = 1
        else:
            word_dict[j] += 1
print('10 most frequent words after stop word removal',Counter(word_dict).most_common(10))

#s = pd.Series(np.concatenate(tweets_df['TokenisedTweetWordRemoval'])).value_counts()
#print(s[:10])

# Q3 Line chart where the horizontal axis corresponds to words and vertical axis indicates the fraction of documents in a which a word appears.

#storing occurence of each word in form of dictionary
doc_freq =defaultdict(int)
for i in tweets_df['TokenisedTweetWordRemoval']:
  for j in set(i):
    doc_freq[j]+=1
doc_freq = dict(sorted(doc_freq.items(), key=lambda item: item[1]))

total_doc = len(tweets_df.TokenisedTweetWordRemoval)

for x in doc_freq:
    doc_freq[x]= (doc_freq[x]/total_doc)

x = list()
for i in range(len(doc_freq)):
  x.append(i)
y= list(doc_freq.values())
plt.plot(x,y)
plt.title("Word frequencies")
plt.xlabel("Word index")
plt.ylabel("Fraction of documents")
plt.show()

# Q4 Multinomial Naive Bayes classifier for the Coronavirus Tweets NLP data set 
corpus_df = pd.read_csv("../data/text_data/Corona_NLP_train.csv",encoding='latin-1')
corpus_original = corpus_df.OriginalTweet.to_numpy()
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2)) 
X = vectorizer.fit_transform(corpus_original)
y = corpus_df['Sentiment']
clf = MultinomialNB()
clf.fit(X,y)
y_pred_class = clf.predict(X)
print('Error rate on original data set ', round(1-accuracy_score(y, y_pred_class),4))

#preprocessing to remove hyperlinks, username, non-alphabetical symbols, extra whitespace, words with length <2 and convert string to lowercase
corpus_df.OriginalTweet = corpus_df.OriginalTweet.str.lower()
corpus_df.OriginalTweet = corpus_df.OriginalTweet.str.replace('((www\.[^\s]+)|(https?://[^\s]+))',' ', regex=True)
corpus_df.OriginalTweet = corpus_df.OriginalTweet.str.replace('@[^\s]+',' ', regex=True)
corpus_df.OriginalTweet = corpus_df.OriginalTweet.str.replace('[^a-zA-Z]', ' ', regex=True)
corpus_df.OriginalTweet = corpus_df.OriginalTweet.str.replace(' +', ' ')
corpus_df['OriginalTweet']= corpus_df['OriginalTweet'].apply(lambda row: str(row).split())
corpus_df['OriginalTweet'] = corpus_df['OriginalTweet'].apply(lambda row: [item for item in row if len(item)>2])

corpus = corpus_df.OriginalTweet

X = vectorizer.fit_transform(corpus.apply(lambda x: ' '.join(x)))
clf = MultinomialNB(alpha=0.5)
clf.fit(X,y)
y_pred_class = clf.predict(X)
print('Error rate after preprocessing ',round(1-accuracy_score(y, y_pred_class),4))
print("--- %s seconds ---" % (time.time() - start_time))