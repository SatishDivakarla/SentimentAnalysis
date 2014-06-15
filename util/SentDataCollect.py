__author__ = 'divakarla'

import glob
import sys
import errno

posTweetsFile = "E:/DataCollection/TestData/posTweets.txt"
negTweetsFile = "E:/DataCollection/TestData/negTweets.txt"
testTweetsFile = "E:/DataCollection/TestData/testTweets.txt"

def readTweets(pos_tweets,neg_tweets,train_tweets,test_tweets):
    posTweets =[]
    negTweets = []
    testTweets =[]

    # Read Positive Tweets to list
    with open(posTweetsFile) as f:
        posTweets = f.read().splitlines()

    # Read Negative Tweets to list
    with open(negTweetsFile) as f:
        negTweets = f.read().splitlines()

    # Read Test Tweets
    with open(testTweetsFile) as f:
        testTweets= (f.read().splitlines())

    for posTweet in posTweets:
        keyValue = []
        keyValue = posTweet,"positive"
        pos_tweets.append(keyValue)

    for negTweet in negTweets:
        keyValue = []
        keyValue = negTweet,"negative"
        neg_tweets.append(keyValue)


    for testTweet in testTweets:
        keyValue = []
        keyValue = testTweet.split('::')
        test_tweets.append(keyValue)

def readTweetsFromFiles(pos_tweets,neg_tweets,train_tweets,test_tweets):
    posTweets =[]
    negTweets = []
    testTweets =[]

    pospath = 'E:/DataCollection/aclImdb/train/pos/*.txt'
    negpath = 'E:/DataCollection/aclImdb/train/neg/*.txt'
    posfiles = glob.glob(pospath)
    negfiles = glob.glob(negpath)

    count =1
    for name in posfiles: # 'file' is a builtin type, 'name' is a less-ambiguous variable name.
        try:
            with open(name) as f: # No need to specify 'r': this is the default.
                posTweets.append(f.read().splitlines())
                count += 1
                if count >50:
                    break
        except IOError as exc:
            if exc.errno != errno.EISDIR: #Do not fail if a directory is found, just ignore it.
                raise # Propagate other kinds of IOError.

    count =1
    for name in negfiles: # 'file' is a builtin type, 'name' is a less-ambiguous variable name.
        try:
            with open(name) as f: # No need to specify 'r': this is the default.
                negTweets.append(f.read().splitlines())
                count += 1
                if count >50:
                    break
        except IOError as exc:
            if exc.errno != errno.EISDIR: #Do not fail if a directory is found, just ignore it.
                raise # Propagate other kinds of IOError.

    for posTweet in posTweets:
        keyValue = []
        keyValue = posTweet,"positive"
        pos_tweets.append(keyValue)

    for negTweet in negTweets:
        keyValue = []
        keyValue = negTweet,"negative"
        neg_tweets.append(keyValue)

    print "Number of Positive tweets : " ,len(pos_tweets)
    print "Number of Negative tweets : " ,len(neg_tweets)

    return
