import argparse
from collections import defaultdict
import numpy as np
import sklearn.metrics as metrics
import random
from scipy import stats
import codecs
parser = argparse.ArgumentParser(description="""Creates baseline predictions using a central value from the training data (average, median, mode) """)
parser.add_argument('train')
parser.add_argument('test')
parser.add_argument("-t","--type", help="one of the next values: average (default), mode, median, random (from the training distribution)", required=False, type=str, default="average") #column index as array indices, starting from 0
args = parser.parse_args()

def getVals(file):
    v = []
    for line in open(file).readlines():
        (idx,val,tweet) = line.strip().split("\t")
        v.append(float(val))
    return np.array(v)

trainvals = getVals(args.train)

centralvals = [0.0] *len(getVals(args.test))


if args.type == "average":
    centralvals = [np.average(trainvals)] * len(centralvals)
elif args.type == "median":
    centralvals = [np.median(trainvals)] * len(centralvals)
elif args.type == "mode": #We cast mode to the discrete case
    centralvals = list(stats.mode([int(v) for v in trainvals],axis=None)[0]) * len(centralvals)
elif args.type == "random":
    random.shuffle(trainvals)
    centralvals = trainvals[:len(centralvals)]
    random.shuffle(centralvals)

for cval, line in zip(centralvals,open(args.test).readlines()):
    a = line.strip().split("\t")
    a[1] = str(cval)
    print "\t".join(a)
