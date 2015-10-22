# -*- coding: utf-8 -*-
import numpy as np
from sklearn import linear_model
import argparse
from sklearn.feature_extraction import DictVectorizer



parser = argparse.ArgumentParser(description="""Uses general sentiment analysys predictions as features for training and testing sentiment analysis for figurative tweets.""")
parser.add_argument('train')
parser.add_argument('test')
args=parser.parse_args()


y_train      = []
train_file          = args.train
test_file    = args.test


test_ids    = []
test_tweets = []

MIN = -5
MAX =  5


gen_sent_file_train = '../../output_general/train.oldgeneralsentimentmodel'
gen_sent_file_test = '../../output_general/trial.oldgeneralsentimentmodel'

def read_general_sent_file(gen_sent_file):
	# skip first line
	# 5 columns:
	# sent_score pos_weight neut_weight neg_weight tweet
	scores = []
	with open(gen_sent_file) as f:
		skip = 0
		for line in f:
			if skip >0:
				line = line.strip()
				line = line.split('\t')[0]
				sent_score,pos_weight,neut_weight,neg_weight = line.split()
				this_score = {'sent_score': float(sent_score),'pos_weight':float(pos_weight), 'neut_weight':float(neut_weight),'neg_weight':float(neg_weight)}
				scores.append(this_score)
			skip+=1

	return scores






# open train file and extract ids, scores, and tweets

with open(train_file,'r') as f:
	for line in f:
		line = line.strip()
		
		id_tag,score,tweet = line.split('\t')
		y_train.append(float(score))

y_train = np.array(y_train)


# open test file and extract ids, scores, and tweets
with open(test_file,'r') as tst:
	for line in tst:
		line = line.strip()
		id_tag,score,tweet = line.split('\t')

		test_ids.append(id_tag)
		test_tweets.append(tweet)


vectorizer = DictVectorizer()
train_gen_sent = read_general_sent_file(gen_sent_file_train)
test_gen_sent  = read_general_sent_file(gen_sent_file_test)

X_train = vectorizer.fit_transform(train_gen_sent)
X_test  = vectorizer.transform(test_gen_sent)




# build model, fit and predict
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

predicted_scores = regr.predict(X_test)

for i in range(len(test_ids)):
	print test_ids[i]+'\t'+str(predicted_scores[i])+'\t'+test_tweets[i]
