from __future__ import print_function
import argparse
import sys
from collections import Counter
import numpy as np
import sklearn.metrics as metrics
from random import random

parser = argparse.ArgumentParser(description="""Internal evaluation scripts""")
parser.add_argument('gold')
parser.add_argument('system')
parser.add_argument('-r','--report',default=0,type=int)

args = parser.parse_args()


def stderrout(*objs):
    print( *objs, file=sys.stderr)


MEAN=-1.99104156234 #TAKEN FROM THE DATA,  UGLY HARDCODED CONSTANT, SORRY
def getVals(file):
    tweets = []
    indices = []
    v = []
    for line in open(file).readlines():
        idx = line.strip().split("\t")[0]
        val = line.strip().split("\t")[1]
        indices.append(idx)
        v.append(float(val))
    return indices, tweets, np.array(v)

indices, tweets,goldvalues = getVals(args.gold)
indicesb, tweets, sysvalues = getVals(args.system)

if indices == indicesb:
    stderrout("match")

prediction_cosine = metrics.pairwise.cosine_similarity(goldvalues,sysvalues)[0][0]
mse = metrics.mean_squared_error(goldvalues,sysvalues)
Rsquare = 1 - (mse / metrics.mean_squared_error(goldvalues,[MEAN]*len(goldvalues)))

out = [args.system, prediction_cosine,mse,Rsquare]
out = [str(s) for s in out]

print("\t".join(out))



if args.report != 0:
    square_errors = np.square(goldvalues - sysvalues)
    C = Counter()
    o = "\t".join(["index","sq_err","gold","system","tweet","tweetindex"])
    stderrout(o)
    for i,v in enumerate(square_errors):
        C[i] = v
    for idx, sq_error in C.most_common(args.report):
        o = "\t".join([str(idx), str(sq_error)[:5], str(goldvalues[idx]),str(sysvalues[idx])[:5],tweets[idx],indices[idx]])
        stderrout(o)

