# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import argparse
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn import linear_model
from functools import partial
from scipy.sparse import *
import sklearn.metrics as metrics




parser = argparse.ArgumentParser(description="""Creates baseline predictions using a BOW representation and a Classification Model.""")
parser.add_argument('train')
parser.add_argument('test')
parser.add_argument("-t","--type", help="one of the next values: count (default), binary, tfidf", required=False, type=str, default="count")
parser.add_argument("-rsw","--removeStopWords",help="remove stop words: True, False (default)", required=False, type=str, default="False")
parser.add_argument("-rht","--removeHashTags",help="remove Hash Tags: True, False (default)", required=False, type=str, default="False")
parser.add_argument("-ngrams","--useNgrams",help="the order of ngrams: two integers, an upper and a lower bound separated by a space. Default is 1 2 , unigrams and bigrams",required=False,type=str,default="1 2")
parser.add_argument("-badvalues","--WhatBadValues",help="what to do with predicted values that go outside the value range of this experiment. 'cap' (default) which brings the value back to a suitable value in the range by setting it to max or min. 'rescale', which rescales the distribution of answers",required=False,type=str,default=None)
parser.add_argument("-classification","--ClassificationMethod",help="the type of classification method used: DT (default), NB, NBM,KNN",required=False, type=str, default="DT")
parser.add_argument("-testScore","--testScores",help="A boolean value that indicates if testScores are available",required=False,type=str,default="False")
args = parser.parse_args()


# given a list of scores assign them to class lables
def to_class_lables(score_list,number_of_classes):
	lables = ['A','B','C','D','E','F','G','H','I','J','K']
	matching_scores_to_lables = [[], [], [], [], [], [], [], [], [], [], []]
	rounded_score_list = [round(x) for x in score_list]
	class_labels=[]



	if number_of_classes == 11:
		all_possible_scores = range(-5,6)
		score_label_dict = dict()
		for i in range(number_of_classes):
			score_label_dict[all_possible_scores[i]] = lables[i]


		for i in range(len(score_list)):
			label = score_label_dict[rounded_score_list[i]]
			lab_index = lables.index(label)
			matching_scores_to_lables[lab_index].append(score_list[i])
			class_labels.append(label)
		categories = lables
	
	else:
		start       = 100/float(number_of_classes)
		edges       = [start*i for i in range(1,number_of_classes+1)]
		
		percentiles = np.percentile(score_list,edges)
		

		categories = lables[:number_of_classes]
		print 'PERCENTILES,:::,',percentiles

		for i in range(len(score_list)):
			score = rounded_score_list[i]
			actual_values_score = score_list[i]
			

			for a in range(number_of_classes):
				if a == 0:
					if score < percentiles[a]:
						label = lables[a]
						matching_scores_to_lables[a].append(actual_values_score)
						#print "score/label: ", str(score) +"/" + str(label)
				elif a >0 and a < number_of_classes-1:
					b = a-1
					if score >= percentiles[b] and score < percentiles[a]:
						label=lables[a]
						matching_scores_to_lables[a].append(actual_values_score)
						#print "score/label: ", str(score) +"/" + str(label)
				elif a == number_of_classes-1:
					b = a-1
					if score>= percentiles[b] and score <= percentiles[a]:
						label = lables[a]
						matching_scores_to_lables[a].append(actual_values_score)
						#print "score/label: ", str(score) +"/" + str(label)

			class_labels.append(label)
	return class_labels,categories,matching_scores_to_lables



def own_tokenizer(sent):
	# input comes in pre-tokenized, and tokens are sepparated by white space
	# this is used in the *Vectorizer functions
	return sent.split(' ')


MIN = -5
MAX =  5


y_train_scores = []
train_ids    = []
y_train      = []
train_tweets = []

train_file          = args.train
test_file           = args.test

vectorizerChoice  		= args.type
removeStops       		= args.removeStopWords
removeHashTags    		= args.removeHashTags
ngram_user_range  		= (int(args.useNgrams.split(' ')[0]), int(args.useNgrams.split(' ')[1]))
bad_values_choice 		=  args.WhatBadValues
classification_type  	= args.ClassificationMethod
testScores_available 	= args.testScores


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
		tweet = remove_user_tags(tweet)

		# if Hash Tags are to be removed
		if removeHashTags == 'True':
			split_tweet = tweet.split(' ')
			wl = [w for w in split_tweet if not w[0] =='#']
			tweet = (' ').join(wl)


		train_ids.append(id_tag)
		y_train_scores.append(float(score))
		train_tweets.append(tweet)


y_true = []
# open test file and extract ids, scores, and tweets
with open(test_file,'r') as tst:
	for line in tst:
		line = line.strip()
		id_tag,score,tweet = line.split('\t')

		#tweet = replace_user_tags(tweet)
		tweet = remove_user_tags(tweet)

		if removeHashTags == 'True':
			split_tweet = tweet.split(' ')
			wl = [w for w in split_tweet if not w[0] =='#']
			tweet = (' ').join(wl)

		test_ids.append(id_tag)
		test_tweets.append(tweet)

		if testScores_available:
			y_true.append(float(score))


#y_true_labels,_,_= to_class_lables(y_true,nclasses)
#y_true_labels   = np.array(y_true_labels)
y_ture = np.array(y_true)


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



 ### what classification model has been chosen:
if classification_type.lower() == 'dt':
 	from sklearn.tree import DecisionTreeClassifier
 	clf = DecisionTreeClassifier()
 	X_train = X_train.todense()
 	X_test  = X_test.todense()

elif classification_type.lower() == 'nb':
 	from sklearn.naive_bayes import GaussianNB
 	clf = GaussianNB()
 	X_train = X_train.todense()
 	X_test  = X_test.todense()

elif classification_type.lower() == 'nbm':
 	from sklearn.naive_bayes import MultinomialNB
 	clf = MultinomialNB()
 	X_train = X_train.todense()
 	X_test  = X_test.todense()

elif classification_type.lower() == 'knn':
 	from sklearn.neighbors import NearestNeighbors
 	# automatically run a way to find the best value of k

 	#clf = NearestNeighbors(n_neighbors=2)



def back_to_numbers(class_scores,scores_to_lables_lists,numclass):
	from scipy.stats import mode
	back_to_values_mean   = np.zeros(len(class_scores))
	back_to_values_mode_small   = np.zeros(len(class_scores))
	back_to_values_mode_larger   = np.zeros(len(class_scores))
	back_to_values_median = np.zeros(len(class_scores))
	back_to_values_max    = np.zeros(len(class_scores))
	back_to_values_min    = np.zeros(len(class_scores))
	lables = ['A','B','C','D','E','F','G','H','I','J','K']
	numbers_lables_dict = dict()
	for j in range(0,11):
		numbers_lables_dict[lables[j]] = j

	for i in range(len(class_scores)):
		cs = class_scores[i]
		bin = numbers_lables_dict[cs]

		back_to_values_mean[i]  		= np.array(scores_to_lables_lists[bin]).mean()
		back_to_values_mode_small[i]   	= mode(scores_to_lables_lists[bin])[0][0]
		back_to_values_mode_larger[i] 	= mode(scores_to_lables_lists[bin])[1][0] 
		back_to_values_median[i] 		= np.median(scores_to_lables_lists[bin])
		back_to_values_max[i]    		= np.array(scores_to_lables_lists[bin]).max()
		back_to_values_min[i]    		= np.array(scores_to_lables_lists[bin]).min()
		
		





	return [back_to_values_mean,back_to_values_mode_small,back_to_values_mode_larger,back_to_values_median,back_to_values_max,back_to_values_min ] 

# loop through all possible number of classes upto 11	
for i in range(2,12):
	nclasses = i
	y_train,categories,current_scores_to_lables = to_class_lables(y_train_scores,nclasses)
	y_train = np.array(y_train)
	

	clf.fit(X_train, y_train)

	predicted_scores = clf.predict(X_test)

	if testScores_available == 'True':
		systypes = ['mean','mode_smaller','mode_larger','meadian','max','min'] 
		systemScores = back_to_numbers(predicted_scores,current_scores_to_lables,nclasses)
		for i  in range(len(systemScores)):
			sysvalues = systemScores[i]
			ss       = systypes[i]
			prediction_cosine = metrics.pairwise.cosine_similarity(y_true,sysvalues)[0][0]
			mse = metrics.mean_squared_error(y_true,sysvalues)
			print '%0.3f , %s , %0.8f,%0.4f' % (nclasses,ss,prediction_cosine,mse)




	"""from sklearn import metrics
	f1score = metrics.f1_score(y_true_labels, predicted_scores)
	#print("f1-score:   %0.3f" % f1score)
	accuracy = metrics.accuracy_score(y_true_labels, predicted_scores)
	#print("Accuracy: %0.3f" % accuracy)
	print "%0.3f \t %0.3f\n" % (accuracy,f1score)
	print("classification report:")
	print(metrics.classification_report(y_true_labels, predicted_scores,target_names=categories))
	print("confusion matrix:")
	print(metrics.confusion_matrix(y_true_labels, predicted_scores,labels=categories))"""



"""
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
"""


