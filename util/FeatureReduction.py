__author__ = 'divakarla'

from nltk.probability import FreqDist, ConditionalFreqDist
from nltk import BigramAssocMeasures
from nltk import BigramCollocationFinder
import SentiUtil


word_fd = FreqDist()
label_word_fd = ConditionalFreqDist()

#TODO: Bigrams are still not considered
def find_1000_best_words(pos_tweet_words,neg_tweet_words, stop_words_filter, bigram_collocation_check):
    for tweet in pos_tweet_words:
        tweet_words  = tweet[0]
        all_words = []
        if bigram_collocation_check:
            bigrams = best_bigram_word_feats(tweet_words)
        if stop_words_filter:
            words = SentiUtil.stopword_filtered_word_feats(tweet_words)
        all_words.extend(words)
        for word in all_words:
            word_fd.inc(word.lower())
            label_word_fd['positive'].inc(word.lower())

    for tweet in neg_tweet_words:
        tweet_words = tweet[0]
        all_words = []
        if bigram_collocation_check:
            bigrams = best_bigram_word_feats(tweet_words)
        if stop_words_filter:
            words = SentiUtil.stopword_filtered_word_feats(tweet_words)
        all_words.extend(words)
        for word in all_words:
            word_fd.inc(word.lower())
            label_word_fd['negative'].inc(word.lower())

    pos_word_count = label_word_fd['positive'].N()
    neg_word_count = label_word_fd['negative'].N()
    total_word_count = pos_word_count + neg_word_count

#    print "Pos word count  : ", pos_word_count
#    print "Neg word count  : ", neg_word_count
#    print "Total word count : " , total_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(label_word_fd['positive'][word],
                                               (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(label_word_fd['negative'][word],
                                               (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
        best = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:10000]
        bestwords = set([w for w, s in best])
    return bestwords

def best_word_feats(words,bestwords):
    newwords =[]
    for word in words:
        if word in bestwords:
            newwords.append(words)
    return newwords


def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return  bigrams
