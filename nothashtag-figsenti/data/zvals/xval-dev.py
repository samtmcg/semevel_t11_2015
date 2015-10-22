#!/usr/bin/env python
import argparse
import math
import sys
from collections import Counter
import numpy as np
import sklearn.metrics as metrics
from sklearn import cross_validation
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import codecs
import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor 
from collections import Counter

def main():
    #goldfile="../twokenized/train+trial.dat"
    goldfile="../twokenized/train.dat"
    goldscores=read_figsenti(goldfile)
    cross=cross_validation.KFold(len(goldscores),n_folds=10,indices=True)
    folder="dev/"
    #folder="final/"
    trainfiles=[x for x in os.listdir(folder) if  x.startswith("train.")]
    for file in trainfiles:
        scores = read_figsenti(folder+"/"+file)
        acc=[]
        for train_index, test_index in cross:
            #print("TRAIN:", len(train_index), "TEST:", len(test_index))
            y_test,y_pred = goldscores[test_index], scores[test_index]
            y_pred=[math.floor(x + 0.5) for x in y_pred]
            y_test=[math.floor(x + 0.5) for x in y_test]
            acc.append(cosine_similarity(y_test,y_pred)[0][0])
            #print acc[-1]
        print "{}\t{}".format(np.mean(acc),file)

def read_figsenti(fname):
    scores=[]
    for line in codecs.open(fname,encoding="utf-8"):
        if line.strip():
            line = line.strip().split("\t")
            value=line[1]
            tid=line[0]
            scores.append(float(value))    
    return np.array(scores)
        
if __name__=="__main__":
    main()
