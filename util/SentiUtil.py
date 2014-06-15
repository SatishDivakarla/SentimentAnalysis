__author__ = 'divakarla'
import nltk
from nltk import BigramAssocMeasures
from nltk import BigramCollocationFinder
from nltk.corpus import stopwords
import itertools

#TODO: Stemming should be supported
def splitWords(tweets):
    newtweets =[]
    for (words, sentiment) in tweets:
        words_filtered = [e.lower() for e in words[0].split() if len(e) >= 3]
        newtweets.append((words_filtered, sentiment))

    return newtweets

'''
def get_words_in_tweets(tweets,stop_words_filter,bigram_collocation_check):
    all_words = []
    for (words, sentiment) in tweets:
        if bigram_collocation_check:
            words  =  bigram_word_feats(words)
        if stop_words_filter:
            words = stopword_filtered_word_feats(words)
        all_words.extend(words)
    return all_words
'''

def bigram_word_feats(tweets, score_fn=BigramAssocMeasures.chi_sq, n=200):
    newwords = []
    all_words = []
    for (words, sentiment) in tweets:
        bigram_finder = BigramCollocationFinder.from_words(words)
        bigrams = bigram_finder.nbest(score_fn, n)
        for ngram in itertools.chain(words, bigrams):
            newwords.append(ngram)

    return newwords

#Method of filtering stop words
def stopword_filtered_word_feats(words):
    newwords = []
    stopset = set(stopwords.words('english'))
    for word in words:
        if word not in stopset:
            newwords.append(word)

    return newwords


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features
