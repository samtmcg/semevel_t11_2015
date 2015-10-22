##### RUN A LEARNING ALGORITHM ON THE  OUTPUT OF FROM MODELS ON TRAINING SET TO LEARN THE SCORE OF THE TEST SET #####
import sys
import subprocess
import numpy as np
import argparse
import os
from to_class_lables_helpers import *
import sklearn.metrics as metrics

import math

parser = argparse.ArgumentParser(description="""stacking.""")
parser.add_argument('scores_folder')
parser.add_argument('training_scores')
parser.add_argument("-model","--LearnignMethod",help="the type of learning method used: DT (default), SVM, randomforrestclass,randomforrestregr,SVR",required=False, type=str, default="DT")
parser.add_argument("-testScore","--testScores",help="A file that has the true scores for the test set",required=False,type=str,default="False")

args = parser.parse_args()


scores_folder          			= args.scores_folder
ytrain_file 					= args.training_scores
learning_model					= args.LearnignMethod
testing_scores 					= args.testScores


training_parts     = [x for x in os.listdir(scores_folder) if  'train.' in x and not x == '.DS_Store']
testing_parts      = [y for y in os.listdir(scores_folder)  if (('trial.' in y) or ('test.' in y) and not y == '.DS_Store')]



classification_models = {'svm','dt'}
###### Some information from the test data
first_address = scores_folder+testing_parts[0]



# find the number of lines (ie number of scores, in the first file in the test data)
p1 = subprocess.Popen(['wc', '-l',first_address], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
out1,err1 = p1.communicate()
out1 = [x for x in out1.split(' ') if x.isdigit()]
test_length = int(out1[0])

## get a list of the tweet ids for test data
p2 = subprocess.Popen(['cut', '-f','1',first_address], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
out2,err2 = p2.communicate()
tweet_ids = [x for x in out2.strip().split('\n')]

###### Some information from the training data
second_address = scores_folder+training_parts[0]
# find the number of lines (ie number of scores, in the first file in the training data)
p1 = subprocess.Popen(['wc', '-l',second_address], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
out1,err1 = p1.communicate()
out1 = [x for x in out1.split(' ') if x.isdigit()]
train_length = int(out1[0])



def get_scores(file_name):
	p = subprocess.Popen(['cut','-f','2',file_name], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
	out, err = p.communicate()
	this_score = np.array([float(x) for x in out.split('\n') if not x == ''])
	

	return this_score


def scores_from_folders(file_name_list,begin_path,expected_length):
	file_scores_dict = dict()
	for file_name in file_name_list: # for every file get the scores column and put into np array
		file_n = begin_path+file_name
		this_score = get_scores(file_n)

		file_type = ('_').join(file_name.split('.')[1:])
		file_scores_dict[file_type] = this_score
		if not len(this_score) == expected_length:
			print 'Error:: Trance to %s' %  file_n
			print 'Error:: number of scores in %s (%s) differ from the others (%s) \n \n' % (file_name,str(len(this_score)),str(expected_length))
			sys.exit()
	return file_scores_dict

import math

def capping(score_list):
	capped_scores = []
	for s in score_list:
		if s <-5:
			s = -5
		elif s > 5:
			s = 5
		capped_scores.append(s)
	return capped_scores

##### geting the training data

training_scores_dictionary = scores_from_folders(training_parts,scores_folder,train_length)
#y_train        = get_scores(ytrain_file)
y_train_scores = get_scores(ytrain_file)

##### geting the test data
test_scores_dictionary  = scores_from_folders(testing_parts,scores_folder,test_length)

##### only use model output from ones available in the test data

n_features = len(test_scores_dictionary.keys())
X_train = np.zeros((train_length,n_features))
X_test  = np.zeros((test_length,n_features))


mod_types = test_scores_dictionary.keys()


#### Make matrices of feature values
for i in range(len(mod_types)):
	#### enter values into training matrix
	enter_train = training_scores_dictionary[mod_types[i]]
	X_train[:,i] = enter_train

	#### enter values into test matrix
	enter_test = test_scores_dictionary[mod_types[i]]
	X_test[:,i] = enter_test


##################################################
 ### what learning model has been chosen:
if learning_model.lower() == 'dt':
 	from sklearn.tree import DecisionTreeClassifier
 	clf = DecisionTreeClassifier()

 	for i in [11]: #range(2,12):
		nclasses = i
		y_train,categories,current_scores_to_lables = to_class_lables(y_train_scores,nclasses)
		y_train = np.array(y_train)


elif learning_model.lower() == 'svm':
	from sklearn import svm
	from sklearn.grid_search import GridSearchCV
	## convert to class lables
	for i in [11]: #range(2,12):
		nclasses = i
		y_train,categories,current_scores_to_lables = to_class_lables(y_train_scores,nclasses)
		y_train = np.array(y_train)
	
	# Set the parameters by grid search and cross-validation
	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
	{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

	clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5, scoring='precision')
	clf.fit(X_train, y_train)
	clf = clf.best_estimator_

elif learning_model.lower() == 'svr':
	from sklearn import svm
	from sklearn.grid_search import GridSearchCV

	# Set the parameters by grid search and cross-validation
	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
	{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

	clf = GridSearchCV(svm.SVR(C=1), tuned_parameters, cv=5)
	clf.fit(X_train, y_train_scores)
	clf = clf.best_estimator_
	y_train = np.array(y_train_scores)



elif learning_model.lower() == 'randomforrestclass':
	from sklearn.ensemble import RandomForestClassifier 
	clf = RandomForestClassifier(n_estimators = 100)
	for i in [11]: #range(2,12):
		nclasses = i
		rounded_scores  = [math.floor(x+0.5) for x in y_train_scores]
		y_train       = capping(rounded_scores)


elif learning_model.lower() == 'randomforrestregr':
	from sklearn.ensemble import RandomForestRegressor
	clf = RandomForestRegressor(n_estimators = 100)
	y_train = y_train_scores


elif learning_model.lower() == 'bayesianridge':
	from sklearn import linear_model
	clf = linear_model.BayesianRidge()
	y_train = y_train_scores

elif learning_model.lower() == 'linearregr':
	from sklearn import linear_model
	clf = linear_model.LinearRegression()
	y_train = y_train_scores

elif learning_model.lower() == 'ridgecv':
	from sklearn import linear_model
	clf = linear_model.RidgeCV()
	y_train = y_train_scores





### fit model and predict
clf.fit(X_train, y_train)
predicted_scores = clf.predict(X_test)
	


##################################################
# If we have scores for the test file
##################################################




if not testing_scores == 'False':
	y_true_float = get_scores(testing_scores)
	rounded_scores  = [math.floor(x+0.5) for x in y_true_float]
	y_true       = capping(rounded_scores)

	if learning_model.lower() in classification_models:
		systypes = ['mean','mode_smaller','mode_larger','meadian','max','min'] 
		systemScores = back_to_numbers(predicted_scores,current_scores_to_lables,nclasses)
		for i  in range(len(systemScores)):
			float_sysvalues = systemScores[i]
			rounded_scores  = [math.floor(x+0.5) for x in float_sysvalues]
			sysvalues       = capping(rounded_scores)

			ss       = systypes[i]
			prediction_cosine = metrics.pairwise.cosine_similarity(y_true,sysvalues)[0][0]
			print '%0.3f , %s , %0.8f' % (nclasses,ss,prediction_cosine)	

	else:
		prediction_cosine = metrics.pairwise.cosine_similarity(y_true,predicted_scores)[0][0]
		mserror			  = metrics.mean_squared_error(y_true,predicted_scores)
		print prediction_cosine, mserror

else: 

	if learning_model.lower() in classification_models:
		systypes = ['mean','mode_smaller','mode_larger','meadian','max','min'] 
		systemScores = back_to_numbers(predicted_scores,current_scores_to_lables,nclasses)
		float_scores = systemScores[-1] # choose the minimum as a way to bring classes back to numbers
		
		rounded_scores = [math.floor(x+0.5) for x in float_scores]
		scores  = capping(rounded_scores)



	else: # we're using a regressor
		float_scores = predicted_scores
		rounded_scores = [math.floor(x+0.5) for x in float_scores]
		scores  = capping(rounded_scores)

		
	for i in range(test_length):
		print str(tweet_ids[i]) + '\t' + str(scores[i])




