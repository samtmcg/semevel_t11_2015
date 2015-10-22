import sys
import subprocess
import numpy as np
import argparse
import os
from to_class_lables_helpers import *
import sklearn.metrics as metrics

import math

def capping_number(s):
	if s <-5:
		s = -5
	elif s > 5:
		s = 5
	return s

parser = argparse.ArgumentParser(description="""special stacking.""")
parser.add_argument('labels_file')
parser.add_argument('general_sent_output')
parser.add_argument('ensemble_output')

args = parser.parse_args()


label_file         			= args.labels_file
gen_sent_file 				= args.general_sent_output
ensemble_file				= args.ensemble_output



"""from the lables file, associate the tweet id with a label,
 if it is 'NOLABEL' then we'll want to use gen_sent score,
  else stacking score
"""
tweet_lab_dict = dict()
with open(label_file) as lf:
	for line in lf:
		line=line.strip()
		line = line.split('\t')
		this_lab = line[0]
		tweet_id = line[1]

		if not this_lab.lower() == 'nolabel':
			tweet_lab_dict[tweet_id] = 'stacking'
		else:
			tweet_lab_dict[tweet_id] = 'gen_sent'


# from gen sent score output associate tweet ids with score

tweet_gensent_score_dict = dict()
with open(gen_sent_file) as gsf:
	for line in gsf:
		line=line.strip()
		line = line.split('\t')

		tweet_id = line[0]
		this_score = line[1]

		tweet_gensent_score_dict[tweet_id] = float(this_score)


# from ensemble score output associate tweet ids with score

tweet_stack_score_dict = dict()
with open(gen_sent_file) as sf:
	for line in sf:
		line=line.strip()
		line = line.split('\t')

		tweet_id = line[0]
		this_score = line[1]

		tweet_stack_score_dict[tweet_id] = float(this_score)


# go through all the tweets and see which score we want to take

all_tweet_ids = tweet_lab_dict.keys()

for ti in all_tweet_ids:
	if tweet_lab_dict[ti] == 'stacking':
		score = tweet_stack_score_dict[ti]
	else:
		score = tweet_gensent_score_dict[ti]


	round_score = math.floor(score+0.5)
	capped_score = capping_number(round_score)

	print ti +'\t' + capped_score


