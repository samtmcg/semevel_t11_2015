# -*- coding: utf-8 -*-
## using brown clusters
#train_file = '../data/twokenized/train.dat'
#bclusters_file = '../data/wordclusters/train+trial+figurativetw0.1.twok-c500.paths'
import numpy as np


def cluster_words_dict(brown_file):
	# 3 columns:
	# 1st: bit string
	# 2nd: word
	# 3rd: count
	with open(brown_file) as f:
		last_bit = '-1'
		clutster_dict = dict()
		word_collection =[]
		for line in f:
			line=line.strip()
			bitstring,word,count = line.split('\t')

			if bitstring == last_bit:
				word_collection.append(word)
				last_bit = bitstring
			else:
				clutster_dict[last_bit] = word_collection
				word_collection = []
				last_bit = bitstring

	return clutster_dict

def make_word_cluster_dictionary(brown_file):
	# 3 columns:
	# 1st: bit string
	# 2nd: word
	# 3rd: count
	cluster_ids = set()
	word_cluster_dict = dict()
	with open(brown_file) as f:
		for line in f:
			line=line.strip()
			bitstring,word,count = line.split('\t')
			word_cluster_dict[word] = bitstring
			cluster_ids.add(bitstring)


	return word_cluster_dict,cluster_ids


def tweets_to_bitstring_cluster_counts(tweets,word_cluster_dictionary):
	from collections import Counter

	count_clusters = []
	for tweet in tweets:
		list_of_bitstrings  = [word_cluster_dictionary[x] if word_cluster_dictionary.has_key(x) else '-1'  for x in tweet]
		cluster_counts      = Counter(list_of_bitstrings)
		count_clusters.append(cluster_counts)
	return count_clusters


