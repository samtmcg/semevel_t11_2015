#!/usr/bin/env python
import sys
import codecs
import argparse
import re
import numpy as np
from scipy import stats
from collections import defaultdict

parser = argparse.ArgumentParser(description="grep for certain patterns, use it as a baseline if test file given")
parser.add_argument('train')
parser.add_argument('-t','--test',default=None)
parser.add_argument('-a','--type',default="average",help="average (default), mode, median")
parser.add_argument('-p','--printlabelTrain',default=False,action="store_true",help="print label of train to stderr")
parser.add_argument('-q','--printlabelTest',default=False,action="store_true",help="print label of train to stderr")
parser.add_argument('-z','--zero',default=False,action="store_true",help="0 for NOLABEL")


def main():

    args = parser.parse_args()

    tweets=defaultdict(list)
    scores=defaultdict(list)
    remaining=0
    for line in codecs.open(args.train):
        line=line.strip()
        _,score,text=line.split("\t")
        subclass=findMatching(text)
        if args.zero and subclass=="NOLABEL":
            score=0.0

        tweets[subclass].append(line)
        scores[subclass].append(float(score))
        
        if args.printlabelTrain:
            print "{}\t{}".format(subclass, line)

    trainscores={}
    for key in tweets.keys():
        #print >>sys.stderr, "=====>",key, np.mean(scores[key])
        trainvals=scores[key]
        if args.type == "average":
            centralval = np.average(trainvals)
        elif args.type == "median":
            centralval = np.median(trainvals)
        elif args.type == "mode": #We cast mode to the discrete case
            centralval = stats.mode([int(v) for v in trainvals],axis=None)[0][0]
        trainscores[key]=centralval
        print >>sys.stderr, "=====>",key, args.type, centralval, "count", len(tweets[key])

    print >>sys.stderr, trainscores
    if args.test:
        print >>sys.stderr, "predicting using {} from train".format(args.type)
        for line in codecs.open(args.test):
            line=line.strip()
            twid,score,text=line.split("\t")
            subclass=findMatching(text)
            print "{}\t{}\t{}".format(twid,trainscores.get(subclass,-1.99104156234),text)
            if args.printlabelTest:
                print >>sys.stderr, "{}\t{}".format(subclass, line)
            

def findMatching(line):
    #### list of regexes ###
    pSarc=re.compile("#sarcas",re.IGNORECASE) # to catch #sarcasm #sarcas #sarcastic #sarcastictweet
    pIron=re.compile("#iron(y|ic)",re.IGNORECASE)
    pNot=re.compile("#not",re.IGNORECASE)
    pLiterally=re.compile(r"\bliterally\b",re.IGNORECASE)
    pVirtually=re.compile(r"\bvirtually\b",re.IGNORECASE)
    pYeahright=re.compile("#yeahright",re.IGNORECASE)
    pOhyoumust=re.compile("Oh.*you must",re.IGNORECASE)
    pAsXas=re.compile(r"\bas .* as\b",re.IGNORECASE)
    pSotospeak=re.compile(r"\bso to speak\b",re.IGNORECASE)
    pDontyoulove=re.compile(r"\bdon't you love\b",re.IGNORECASE)
    pProverbial=re.compile(r"\bproverbial\b",re.IGNORECASE)
    pJustkidding=re.compile("#justkidding",re.IGNORECASE)
    pNot2=re.compile(r"\bnot\b",re.IGNORECASE)
    pAbout=re.compile(r"\babout\b",re.IGNORECASE)
    pOh=re.compile(r"\boh\b",re.IGNORECASE)

    DEFAULT="NOLABEL"
    if pSarc.search(line):
        return "sarcasm"
    elif pIron.search(line):
        return "iron"
    elif pNot.search(line):
        return "not"
    elif pLiterally.search(line):
        return "literally"
    elif pVirtually.search(line):
        return "virtually"
    elif pYeahright.search(line):
        return "yeahright"
    elif pOhyoumust.search(line):
        return "ohyoumust"
    elif pAsXas.search(line):
        return "asXas"
    elif pSotospeak.search(line):
        return "sotospeak"
    elif pDontyoulove.search(line):
        return "dontyoulove"
    elif pProverbial.search(line):
        return "proverbial"
    elif pJustkidding.search(line):
        return "justkidding"
    elif pNot2.search(line):
        return "not2"
    elif pAbout.search(line):
        return "about"
    elif pOh.search(line):
        return "oh"
    else:
        return DEFAULT


if __name__=="__main__":
    main()


