import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage
"""
python distributions file_with_scores1 name1 file_with_scores2 name2
"""

text = sys.argv[1]
name = sys.argv[2]
text2= sys.argv[3]
name2= sys.argv[4]

def get_score(txt):
	scores = []
	with open(txt) as f:
		for line in f:
			line = line.strip()
			#id,score,tweet = line.split('\t')
			stuff = line.split('\t')
			score = stuff[1]

			scores.append(float(score))

	return scores


scores1 = np.array(get_score(text))
scores2 = np.array(get_score(text2))

bins = np.linspace(-5,5,11)
plt.hist(scores1,bins,color='blue',label=name)
plt.hist(scores2,bins,color='black',alpha=0.7,label=name2)
plt.xlim([-5,5])
plt.xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
title = 'Distribution of Scores: %s and %s' % (name, name2)
plt.title(title)
plt.xlabel("Prediction Value")
plt.ylabel("Number of Predictions")
plt.legend()
plt.show()


