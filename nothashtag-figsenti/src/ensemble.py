##### READ IN RESULTS FROM THE OUTPUT OF MODELS TO CREATE AN AVERAGE OR WEIGHTED AVERAGE #####
# $ python ensemble.py folder_where_model_predicitons_are path_to_scored_models_file

import sys
import subprocess
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="""Creates average ensemble prediction outputs from models and the cosine similarity scores,stored together in a csv file.""")
parser.add_argument('model_outputs_folder')
parser.add_argument('model_scores_file')
parser.add_argument("-t","--type", help="type of average, mean (default) or weight", required=False, type=str, default="mean")
args = parser.parse_args()


outputs_files           = args.model_outputs_folder
file_with_weights       = args.model_scores_file
ens_type 				= args.type

parts     = [x for x in os.listdir(outputs_files) if not x == '.DS_Store']




first_address = outputs_files+parts[0]
# find the number of lines (ie number of scores, in the first file)
p1 = subprocess.Popen(['wc', '-l',first_address], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
out1,err1 = p1.communicate()
out1 = [x for x in out1.split(' ') if x.isdigit()]
length = int(out1[0])

## get a list of the tweet ids
p2 = subprocess.Popen(['cut', '-f','1',first_address], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
out2,err2 = p2.communicate()
tweet_ids = [x for x in out2.strip().split('\n')]

## get a list of the tweets
p3 = subprocess.Popen(['cut', '-f','3',first_address], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
out3,err3 = p3.communicate()
tweets = [x for x in out3.strip().split('\n')]



all_scores = np.zeros(length)
file_scores_dict = dict()
for file_name in parts: # for every file get the scores column and put into np array
	file_n = outputs_files+file_name
	p = subprocess.Popen(['cut','-f','2',file_n], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
	out, err = p.communicate()
	this_score = np.array([float(x) for x in out.split('\n') if not x == ''])
	file_scores_dict[file_name] = this_score # store these scores in a dictionary indexed by the name of the results file
	if len(this_score) == length:
		all_scores += this_score
	else:
		print 'Error:: number of scores in %s (%s) differ from the others (%s) \n \n' % (file_name,str(len(this_score)),str(length))

		sys.exit()




running_total = np.zeros(length)
total_weights = 0
if ens_type.lower() == 'mean':
	scores = all_scores / float(len(parts))


if ens_type.lower() == 'weight' or ens_type.lower() == 'w':
	# create a dictionary of the model_file_outputs and their cosine similarity
	model_score_dict = dict()
	
	with open(file_with_weights,'r') as weightFile:
		for line in weightFile:
			line = line.strip()
			model_name,score = line.split(',')
			model_score_dict[model_name] = float(score)

	for file_key in file_scores_dict.keys():
		current_scores = file_scores_dict[file_key]
		current_weight = model_score_dict[file_key]

		current_weighted_scores = current_scores * current_weight
		
		running_total += current_weighted_scores
		total_weights += current_weight

	scores = running_total / total_weights




for i in range(length):
	print str(tweet_ids[i]) + '\t' + str(scores[i]) + '\t' + tweets[i]




