# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import argparse
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn import linear_model
from functools import partial
from scipy.sparse import *




parser = argparse.ArgumentParser(description="""Creates baseline predictions using a BOW representation and a Linear Regrssion Model.""")
parser.add_argument('train')
parser.add_argument('test')
parser.add_argument("-t","--type", help="one of the next values: count (default), binary, tfidf", required=False, type=str, default="count")
parser.add_argument("-rsw","--removeStopWords",help="remove stop words: True, False (default)", required=False, type=str, default="False")
parser.add_argument("-rht","--removeHashTags",help="remove Hash Tags: True, False (default)", required=False, type=str, default="False")
parser.add_argument("-ngrams","--useNgrams",help="the order of ngrams: two integers, an upper and a lower bound separated by a space. Default is 1 2 , unigrams and bigrams",required=False,type=str,default="1 2")
parser.add_argument("-badvalues","--WhatBadValues",help="what to do with predicted values that go outside the value range of this experiment. 'cap' (default) which brings the value back to a suitable value in the range by setting it to max or min. 'rescale', which rescales the distribution of answers",required=False,type=str,default=None)
args = parser.parse_args()


def own_tokenizer(sent):
	# input comes in pre-tokenized, and tokens are sepparated by white space
	# this is used in the *Vectorizer functions
	return sent.split(' ')


MIN = -5
MAX =  5

train_ids    = []
y_train      = []
train_tweets = []

train_file          = args.train
test_file           = args.test

vectorizerChoice  = args.type
removeStops       = args.removeStopWords
removeHashTags    = args.removeHashTags
ngram_user_range  = (int(args.useNgrams.split(' ')[0]), int(args.useNgrams.split(' ')[1]))
bad_values_choice =  args.WhatBadValues


test_ids    = []
test_tweets = []

def replace_user_tags(tweet):
	# removes references to other users, but replaces with a special token,
	# so does not remove the fact that they do reference others
	split_tweet = tweet.split(' ')
	nameless_tweet=[]
	for w in split_tweet:
		if w[0] == '@':
			nameless_tweet.append('referenceAnotherUser')
		else:
			nameless_tweet.append(w)
	fixed_tweet = (' ').join(nameless_tweet)
	return fixed_tweet

def remove_user_tags(tweet):
	# removes references to other users
	split_tweet = tweet.split(' ')
	nameless_tweet=[]
	for w in split_tweet:
		if not w[0] == '@':
			nameless_tweet.append(w)

	fixed_tweet = (' ').join(nameless_tweet)
	return fixed_tweet



# open train file and extract ids, scores, and tweets

with open(train_file,'r') as f:
	for line in f:
		line = line.strip()
		
		id_tag,score,tweet = line.split('\t')


		### want to remove references to other twitter users, without removing the fact that they references a user
		#tweet = replace_user_tags(tweet)
		#tweet = remove_user_tags(tweet)

		# if Hash Tags are to be removed
		if removeHashTags == 'True':
			split_tweet = tweet.split(' ')
			wl = [w for w in split_tweet if not w[0] =='#']
			tweet = (' ').join(wl)


		train_ids.append(id_tag)
		y_train.append(float(score))
		train_tweets.append(tweet)

y_train = np.array(y_train)

# open test file and extract ids, scores, and tweets
with open(test_file,'r') as tst:
	for line in tst:
		line = line.strip()
		id_tag,score,tweet = line.split('\t')

		tweet = replace_user_tags(tweet)

		if removeHashTags == 'True':
			split_tweet = tweet.split(' ')
			wl = [w for w in split_tweet if not w[0] =='#']
			tweet = (' ').join(wl)

		test_ids.append(id_tag)
		test_tweets.append(tweet)





# different possible BOW representations:

# remove stopwords or not? Just using built in list of english stopwords
if removeStops == 'True':
	removeStopwords = 'english'
else:
	removeStopwords = None

if vectorizerChoice == 'count':
	vect_model = CountVectorizer(tokenizer=own_tokenizer,lowercase=False,binary=False,stop_words=removeStopwords,ngram_range=ngram_user_range)

elif vectorizerChoice == 'binary':
	vect_model = CountVectorizer(tokenizer=own_tokenizer,lowercase=False,binary=True,stop_words=removeStopwords,ngram_range=ngram_user_range)

elif vectorizerChoice == 'tfidf':
	vect_model = TfidfVectorizer(tokenizer=own_tokenizer,lowercase=False,binary=False,stop_words=removeStopwords,ngram_range=ngram_user_range)




# transform tweets to vector space representation
X_train    = vect_model.fit_transform(train_tweets)
X_test     = vect_model.transform(test_tweets)







# build model, fit and predict
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

predicted_scores = regr.predict(X_test)

if bad_values_choice == 'cap':

	for i in range(len(predicted_scores)):

		this_score = predicted_scores[i]
		
		if this_score > MAX:
			this_score = MAX

		elif this_score <MIN:
			this_score = MIN

		predicted_scores[i] = this_score

elif bad_values_choice == 'rescale':
	# rescale the distribution
	bad_values_choice = 'rescale'

if not (len(test_ids) == len(predicted_scores)) and (len(test_ids)==len(test_tweets)):
	print "ERROR:: lost data in test\n"
	print 'Number of test_ids: \t', len(test_ids)
	print 'Number of predicted_scores: \t', len(predicted_scores)
	print 'Number of test tweets: \t', len(test_tweets)
else:
	for i in range(len(test_ids)):
		print test_ids[i]+'\t'+str(predicted_scores[i])+'\t'+test_tweets[i]
	#print 'weights: '
	#features = pd.Series(regr.coef_, index=vect_model.get_feature_names())
	#importance_order = features.abs().order(ascending=False).index
	#for i in range(300):
		#s = features[importance_order].index[i] + ' ' + str(features[importance_order].ix[i]) + '\n'
		#sys.stdout.write(s.encode('utf-8'))



