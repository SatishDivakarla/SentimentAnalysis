from SentimentAnalysis.util import SentDataCollect

__author__ = 'divakarla'

import nltk
import collections
from nltk import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk import BigramAssocMeasures
from nltk import BigramCollocationFinder
from nltk.probability import FreqDist, ConditionalFreqDist

import itertools

pos_tweets = []
neg_tweets = []
train_tweets = []
test_tweets = []

#SentDataCollect.readTweets(pos_tweets,neg_tweets,train_tweets,test_tweets)
SentDataCollect.readTweetsFromFiles(pos_tweets,neg_tweets,train_tweets,test_tweets)


def printTweetsForDebugging():
    print "##########################################"
    print  "\n Positive Tweets \n"
    print pos_tweets
    print "##########################################"
    print  " \n Negative Tweets \n"
    print neg_tweets
    print "##########################################"
    print  " \n Test Tweets \n"
    print test_tweets

def splitWords(tweets):
    newtweets =[]
    for (words, sentiment) in tweets:
        words_filtered = [e.lower() for e in words[0].split() if len(e) >= 3]
        newtweets.append((words_filtered, sentiment))
    return newtweets

def get_words_in_tweets(tweets,stop_words_filter,bigram_collocation_check):
    all_words = []
    for (words, sentiment) in tweets:
        if bigram_collocation_check:
            words  =  bigram_word_feats(words)
        if stop_words_filter:
            words = stopword_filtered_word_feats(words)
        all_words.extend(words)
    return all_words

def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    newwords = []
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


# Extract relevant features from the document
def extract_features(document):
    document_words = set(document)
    features = {}

    for word in word_features:
        if(len(word) == 2):
            newword  = word[0] + " " +  word[1]
            features['contains(%s)' % newword] = (newword in document_words)
        else:
            features['contains(%s)' % word] = (word in document_words)
    return features

def word_feats(words):
    return dict([(word, True) for word in words])

#printTweetsForDebugging()

# Merge Pos and Neg tweets and split the words
pos_train_tweets = splitWords(pos_tweets)
neg_train_tweets = splitWords(neg_tweets)

print "Cutoffs... 75-25"
pos_cut_off = len(pos_train_tweets) *3/4
neg_cut_off = len(neg_train_tweets) *3/4

train_tweets = neg_train_tweets[:neg_cut_off] + pos_train_tweets[:pos_cut_off]
test_tweets =  neg_train_tweets[neg_cut_off:] + pos_train_tweets[pos_cut_off:]

print 'Training  on %d instances, test on %d instances' % (len(train_tweets), len(test_tweets))


### Classifier

# List of word features from tweets
words_in_tweets = get_words_in_tweets(train_tweets,True,True)
word_features = get_word_features(words_in_tweets)

# Apply features
print " Applying features for training set..."
training_set = nltk.classify.util.apply_features(extract_features, train_tweets)

print " Applying features for test set ..."
test_set = nltk.classify.util.apply_features(extract_features, test_tweets)

print "Training features count = %d" %(len(training_set))
print "Test features count = %d" %(len(test_set))
# Training
print "Begin training....."
#classifier = nltk.NaiveBayesClassifier.train(training_set)
classifier = NaiveBayesClassifier.train(training_set)
print "Test Set Features count"
print len(test_set)
print 'accuracy:', nltk.classify.util.accuracy(classifier, test_set)
print classifier.show_most_informative_features(100)


'''
#Testing
print "Testing  ...."
for tweet in test_tweets:
    print tweet[0]
    test_extracted_features = extract_features(tweet[0].split())
    observed = classifier.classify(test_extracted_features)
'''

print "Begin collecting stats..."
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
for i, (tweets, label) in enumerate(test_set):
    refsets[label].add(i)
    #test_extracted_features = extract_features(tweets.split())
    observed = classifier.classify(tweets)
    testsets[observed].add(i)


print "Begin metrics check..."
print 'pos precision:', nltk.metrics.precision(refsets['positive'], testsets['positive'])
print 'pos recall:', nltk.metrics.recall(refsets['positive'], testsets['positive'])
print 'pos F-measure:', nltk.metrics.f_measure(refsets['positive'], testsets['positive'])
print 'neg precision:', nltk.metrics.precision(refsets['negative'], testsets['negative'])
print 'neg recall:', nltk.metrics.recall(refsets['negative'], testsets['negative'])
print 'neg F-measure:', nltk.metrics.f_measure(refsets['negative'], testsets['negative'])

word_fd = FreqDist()
label_word_fd = ConditionalFreqDist()

for tweet in pos_train_tweets:
    tweet_words  = tweet[0]
    pos_tweet_words_without_stop =stopword_filtered_word_feats(tweet_words)
    for word in pos_tweet_words_without_stop:
        word_fd.inc(word.lower())
        label_word_fd['positive'].inc(word.lower())

for tweet in neg_train_tweets:
    tweet_words  = tweet[0]
    neg_tweet_words_without_stop =stopword_filtered_word_feats(tweet_words)
    for word in neg_tweet_words_without_stop:
        word_fd.inc(word.lower())
        label_word_fd['negative'].inc(word.lower())

pos_word_count = label_word_fd['positive'].N()
neg_word_count = label_word_fd['negative'].N()
total_word_count = pos_word_count + neg_word_count

print "Pos word count  : ", pos_word_count
print "Neg word count  : ", neg_word_count
print "Total word count : " , total_word_count

word_scores = {}

for word, freq in word_fd.iteritems():
    pos_score = BigramAssocMeasures.chi_sq(label_word_fd['positive'][word],
        (freq, pos_word_count), total_word_count)
    neg_score = BigramAssocMeasures.chi_sq(label_word_fd['negative'][word],
        (freq, neg_word_count), total_word_count)
    word_scores[word] = pos_score + neg_score

best = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:10000]
bestwords = set([w for w, s in best])

def best_word_feats(words):
    newwords =[]
    for word in words:
        if word in bestwords:
            newwords.append(words)
    return newwords

word_features = bestwords

# Apply features
print " Applying features for training set..."
training_set = nltk.classify.util.apply_features(extract_features, train_tweets)

print " Applying features for test set ..."
test_set = nltk.classify.util.apply_features(extract_features, test_tweets)

print "Training features count = %d" %(len(training_set))
print "Test features count = %d" %(len(test_set))
# Training
print "Begin training....."
#classifier = nltk.NaiveBayesClassifier.train(training_set)
classifier = NaiveBayesClassifier.train(training_set)
print "Test Set Features count"
print len(test_set)
print 'accuracy:', nltk.classify.util.accuracy(classifier, test_set)
print classifier.show_most_informative_features(100)


'''
#Testing
print "Testing  ...."
for tweet in test_tweets:
    print tweet[0]
    test_extracted_features = extract_features(tweet[0].split())
    observed = classifier.classify(test_extracted_features)
'''

print "Begin collecting stats..."
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
for i, (tweets, label) in enumerate(test_set):
    refsets[label].add(i)
    #test_extracted_features = extract_features(tweets.split())
    observed = classifier.classify(tweets)
    testsets[observed].add(i)


print "Begin metrics check..."
print 'pos precision:', nltk.metrics.precision(refsets['positive'], testsets['positive'])
print 'pos recall:', nltk.metrics.recall(refsets['positive'], testsets['positive'])
print 'pos F-measure:', nltk.metrics.f_measure(refsets['positive'], testsets['positive'])
print 'neg precision:', nltk.metrics.precision(refsets['negative'], testsets['negative'])
print 'neg recall:', nltk.metrics.recall(refsets['negative'], testsets['negative'])
print 'neg F-measure:', nltk.metrics.f_measure(refsets['negative'], testsets['negative'])

'''
print "Testing ..."

tweet = 'I admit, the great majority of films released before say 1933 are just not for me. Of the dozen or so "major" silents I have viewed, one I loved (The Crowd), and two were very good (The Last Command and City Lights, that latter Chaplin circa 1931).<br /><br />So I was apprehensive about this one, and humor is often difficult to appreciate (uh, enjoy) decades later. I did like the lead actors, but thought little of the film.<br /><br />One intriguing sequence. Early on, the guys are supposed to get "de-loused" and for about three minutes, fully dressed, do some schtick. In the background, perhaps three dozen men pass by, all naked, white and black (WWI ?), and for most, their butts, part or full backside, are shown. Was this an early variation of beefcake courtesy of Howard Hughes?'
test_extracted_features = extract_features(tweet.split())
print test_extracted_features
observed = classifier.classify(test_extracted_features)
print observed
'''

'''
#Accuracy
print "Accuracy Check"
test_tweets = splitWords(test_tweets)
print "Test tweets after word splits......"
print test_tweets
test_set = nltk.classify.util.apply_features(extract_features, test_tweets)
total_set = training_set + test_set

print "Total Set  ...  "
print len(total_set)
print nltk.classify.util.accuracy(classifier,total_set)
#print classifier.show_most_informative_features()

'''