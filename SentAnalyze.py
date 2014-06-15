from SentimentAnalysis.util import SentDataCollect

__author__ = 'divakarla'


import nltk

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
        words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
        newtweets.append((words_filtered, sentiment))
    return newtweets

#print neg_tweets

#printTweetsForDebugging()

# Merge Pos and Neg tweets and split the words
train_tweets = splitWords(pos_tweets + neg_tweets)

#print "Train tweets after word splits......"
#print train_tweets

# Construct same for test tweets
#test_tweets = splitWords(test_tweets)
#print "Test tweets after word splits......"
#print test_tweets


### Classifier


# List of word features from tweets
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

words_in_tweets = get_words_in_tweets(train_tweets)
word_features = get_word_features(words_in_tweets)

# Extract relevant features from the document
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


'''
document = ['love', 'this', 'car']

extracted_features  = extract_features(document)
print "Extracted Features are : "
print extracted_features
print "Extracted Features size is : "
print len(extracted_features)
'''


# Apply features
training_set = nltk.classify.util.apply_features(extract_features, train_tweets)

#print "Training set is : "
#print training_set

# Training
classifier = nltk.NaiveBayesClassifier.train(training_set)

'''
#Testing
print "Testing  ...."
for tweet in test_tweets:
    print tweet[0]
    test_extracted_features = extract_features(tweet[0].split())
    observed = classifier.classify(test_extracted_features)
'''
print "Testing ..."

tweet = 'I admit, the great majority of films released before say 1933 are just not for me. Of the dozen or so "major" silents I have viewed, one I loved (The Crowd), and two were very good (The Last Command and City Lights, that latter Chaplin circa 1931).<br /><br />So I was apprehensive about this one, and humor is often difficult to appreciate (uh, enjoy) decades later. I did like the lead actors, but thought little of the film.<br /><br />One intriguing sequence. Early on, the guys are supposed to get "de-loused" and for about three minutes, fully dressed, do some schtick. In the background, perhaps three dozen men pass by, all naked, white and black (WWI ?), and for most, their butts, part or full backside, are shown. Was this an early variation of beefcake courtesy of Howard Hughes?'
test_extracted_features = extract_features(tweet.split())
print test_extracted_features
observed = classifier.classify(test_extracted_features)
print observed

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