__author__ = 'divakarla'

import nltk
import collections
from nltk import NaiveBayesClassifier
from util import SentiUtil
from util import FeatureReduction
from util import SentDataCollect

import sys
import pickle

pos_tweets = []
neg_tweets = []
train_tweets = []
test_tweets = []

# Extract relevant features from the document
def extract_features(document):
    document_words = set(document)
    features = {}

    for word in word_features:
        if(len(word) == 2):
            newword = word[0] + " " + word[1]
            if (newword in document_words) == True:
                features['contains(%s)' % newword] = True
        else:
            if (word in document_words) == True:
                features['contains(%s)' % word] = True
    return features

total = len(sys.argv)
if total == 1:
    #SentDataCollect.readTweets(pos_tweets,neg_tweets,train_tweets,test_tweets)
    SentDataCollect.readTweetsFromFiles(pos_tweets,neg_tweets,train_tweets,test_tweets)
    # Merge Pos and Neg tweets and split the words
    pos_tweet_words = SentiUtil.splitWords(pos_tweets)
    neg_tweet_words = SentiUtil.splitWords(neg_tweets)

    print "Cutoffs... 75-25"
    pos_cut_off = len(pos_tweet_words) *3/4
    neg_cut_off = len(neg_tweet_words) *3/4
    train_tweets = neg_tweet_words[:neg_cut_off] + pos_tweet_words[:pos_cut_off]
    test_tweets =  neg_tweet_words[neg_cut_off:] + pos_tweet_words[pos_cut_off:]
    print 'Training  on %d instances, test on %d instances' % (len(train_tweets), len(test_tweets))

    # List of word features from tweets
    # stop_words_filter removes the stop words
    # bigram_collocation_check considers the bigrams

    #words_in_tweets = SentiUtil.get_words_in_tweets(train_tweets,stop_words_filter=True,bigram_collocation_check=True)
    # Reducing features
    word_features = FeatureReduction.find_1000_best_words(pos_tweet_words,neg_tweet_words,stop_words_filter=True,bigram_collocation_check=True)
    wordfeatures_file = open('word_features.txt', 'w')
    for item in word_features:
        wordfeatures_file.write("%s\n" % item)

    print "After reducing features, no of word features are : " , len(word_features)

    # Apply features
    print " Applying features for training set..."
    training_set = nltk.classify.util.apply_features(extract_features, train_tweets)
    print " Applying features for test set ..."
    test_set = nltk.classify.util.apply_features(extract_features, test_tweets)
    print "Training features count = %d" %(len(training_set))
    print "Test features count = %d" %(len(test_set))
    # Training
    print "Begin training....."
    classifier = NaiveBayesClassifier.train(training_set)
    print "Saving classifier..."
    f = open('naive_bayes.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()
    print 'accuracy:', nltk.classify.util.accuracy(classifier, test_set)
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
    print "End collecting stats..."

elif total == 2:
    #wordfeatures_file = open('word_features.txt', 'r')
    #word_features = wordfeatures_file.readlines()
    filename = 'word_features.txt'
    word_features = open(filename).read().splitlines()
    #print word_features
    print "After reading from file : " , len(word_features)
    cmdargs = str(sys.argv)
    classifier_name  =  cmdargs[0]
    f= open('naive_bayes.pickle')
    classifier = pickle.load(f)
    f.close()
else:
    print "Wrong number of arguments.."
    exit()

classifier.show_most_informative_features(10)
Flag = False
while Flag == False:
    input_recieved = raw_input("Do you want to test more?")
    if input_recieved.upper() == "YES":
        sentence  =  raw_input("Please enter a sentence")
        test_extracted_features = extract_features(sentence.split())
        if len(test_extracted_features) == 0:
            print "No features identified from statement. System may need more training."
        else:
            print test_extracted_features
            observed = classifier.classify(test_extracted_features)
            print observed
        Flag = False
    elif input_recieved.upper() == "NO":
        print "Thanks for using our tool."
        Flag = True
    else:
        print "Sorry I didn't get you !!"