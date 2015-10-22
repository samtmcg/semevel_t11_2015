import os
import sys
import subprocess
import math

file_name = '/users/sarahmcgillion/Desktop/test_sorted_gsa'


with open(file_name) as fn:
	for line in fn:
		line = line.strip()
		line = line.split('\t')
		tweet_id = line[0]
		val		= float(line[1])

		rounded_val = math.floor(val+0.5)

		if rounded_val<-5:
			score = -5
		elif rounded_val >5:
			score = 5
		else:
			score = rounded_val

		print tweet_id +'\t' + str(score)