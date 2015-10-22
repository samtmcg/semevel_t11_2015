__author__ = 'alonso'
# -*- coding: utf-8 -*-
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="""Takes input like the one provided by the threeway sentiment analysis predictor and maps it to a -5,5 values.""")
parser.add_argument('input')
parser.add_argument('filewithkeys')

args = parser.parse_args()
POSITIVE=5.0
NEGATIVE=-5.0
NEUTRAL=0.0

#0 0.00242768 0.364046 0.633526	My lips are about as dry as a black guys elbows right now ."

for line, idline in zip(open(args.input).readlines()[1:],open(args.filewithkeys).readlines()):
    tweet= line.strip().split("\t")[1]
    idx = idline.split("\t")[0]

    pred_class, pos, neg, neut = line.strip().split("\t")[0].strip().split(" ")

    outval = np.average([POSITIVE, NEGATIVE, NEUTRAL], weights=[float(pos),float(neg),float(neut)])
    print "\t".join([str(idx), str(outval)])