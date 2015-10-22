#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import codecs
import sys
import numpy as np
import re
import math
import preproc
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression, RidgeCV, Lasso, ElasticNet, BayesianRidge
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

from sklearn import cross_validation
from sklearn.pipeline import FeatureUnion
from collections import Counter
import pandas as pd
import pylab as plt

from sklearn import decomposition
from sklearn.mixture import GMM

def getFeatures(inputd,cluster=None,noGrep=False,ngram="1-2-3",gmm=None):
    features=[]
    if cluster:
        clusterids=set(cluster.values())
    print gmm
    for i, tweet in enumerate(inputd):
        d={}
        tweet="<S> "+tweet+" </S>"
        words=tweet.split(" ")
        words = [w for w in words if w not in ENGLISH_STOP_WORDS]
        # ngram features (G)
        for i in range(len(words)) :
            # up to N n-grams
            for n in ngram.split("-"):
                gram = "G "
                N=int(n)
                for j in range(i,min(i+N, len(words))) :
                    gram += words[j] +  " "
                    if len(gram.split(" "))==N+2:
                        d[gram]=1 #binary
            
        # number of ONLY upcased words (U)
        d["W upper"] = sum(all(c.isupper() for c in w) for w in words)

        # punctuation:
        # p1: the number of contiguous sequences of
        # exclamation marks, p2: question marks, and
        # p3: both exclamation and question marks;
        # p4: ? or ! in last token
        d["P P!"] = len(re.findall('[!]+',tweet))
        d["P P?"] = len(re.findall('[?]+',tweet))
        d["P P!?"] = len(re.findall('[!?]*(!+\?+)[!?]*|[!?]*(\?+!+)[!?]*',tweet))
        d["P Pl!?"] = len(re.findall('[!?]+',tweet))

        ### gmm 
        try:
            values = gmm[i]
            for idx, v in enumerate(values):
                d["gmm{}".format(idx)]=v 
        except:
            donotadd=1

        # Brown clusters (B)
        # presence or absence of tokens in cluster
        if cluster:
            active_clusters = [cluster[w] for w in words if w in cluster]
            for c in clusterids:
                if c in active_clusters:
                    d["B "+c]=1

#         wordsLower = [x.lower() for x in words]
#          # skip gram features (S) on lower-cased words
#         for i in range(len(wordsLower)) :
#             if i+2 < len(wordsLower) :
#                 gram="S " + wordsLower[i] + " * " + wordsLower[i+2]
#                 d[gram]=d.get(gram,0)+1
#             if i+3 < len(wordsLower) :
#                 gram="S " + wordsLower[i] + " * " + wordsLower[i+2]+ " " + wordsLower[i+3] 
#                 d[gram]=d.get(gram,0)+1
#                 gram="S " + wordsLower[i] + " " + wordsLower[i+1]+ " * " + wordsLower[i+3] 
#                 d[gram]=d.get(gram,0)+1

        if not noGrep:
            # "grep label"
            d['label']=findMatching(tweet)

        features.append(d)

    return features
        

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


def main():

    parser = argparse.ArgumentParser(description="Ridge regression model")
    parser.add_argument('train', help="train data")
    parser.add_argument('test', help="test data")
    parser.add_argument('--seed', help="random seed",type=int,default=987654321)
    parser.add_argument('--debug', help="debug", action='store_true', default=False)
    parser.add_argument('--removeHashTags', help="removeHashTags", action='store_true', default=False)
    parser.add_argument('--cv', help="run cross-validation", action='store_true', default=False)
    parser.add_argument('--cvp', help="output Lcv scores", action='store_true', default=False)
    parser.add_argument('--alpha', help="alpha parameter", type=float, default=1)
    parser.add_argument('--classweight', help="class weight for +/- classifier", type=float, default=22)
    parser.add_argument('--out', help="output gold and predictions", action='store_true', default=False)
    parser.add_argument('--compare', help="compare to linReg", action='store_true', default=False)
    parser.add_argument('--pred', help="output predictions", action='store_true', default=False)
    parser.add_argument('--plot', help="plot predictions", action='store_true', default=False)
    parser.add_argument('--noGrep', help="no label feats", action='store_true', default=False)
    parser.add_argument('--cluster', help="brown clusters", type=str)
    parser.add_argument('--ngram', help="n-grams", type=str,default="1-2-3")
    parser.add_argument('-c','--components', help="num PCA components", type=int, default=100)
    parser.add_argument('-g','--gmmcomponents', help="num GMM components", type=int, default=12)
    parser.add_argument('--gmm', help="add gmm features", action='store_true', default=False)

    args = parser.parse_args()

    
    ####### load data
    tweetids,inputd,labels=read_data_file(args.train,args.removeHashTags)
    tweetids=np.array(tweetids)
    print >>sys.stderr, "{} instances loaded.".format(len(inputd))
    print >>sys.stderr, u"Instance: {} {}".format(labels[-2],inputd[-2])


    np.random.seed(args.seed)
    
    data ={}

    print >>sys.stderr, len(labels)
    # target labels



    data['target'] = np.array(labels)

    vectorizer=DictVectorizer()
    vectorizernogmm=DictVectorizer()
    cluster=None
    
    if args.cluster:
        word2clusters = {}
        for l in map(str.strip,open(args.cluster).readlines()) :
            bitstring,word,count = l.split("\t")
            word2clusters[word] = bitstring
        cluster=word2clusters


    gmm_predicted=[]
    if args.gmm:
        ## get features for gmm
        X_train_dict = getFeatures(inputd,cluster=cluster,noGrep=args.noGrep,ngram=args.ngram)
        X_train = vectorizernogmm.fit_transform(X_train_dict)

        pca = decomposition.RandomizedPCA(n_components=args.components)
        X_train_pca = pca.fit_transform(X_train)
        print X_train_pca
        gmm = GMM(n_components = args.gmmcomponents,
                  covariance_type = "full",
                  min_covar = 0.01)
        
        gmm.fit(X_train_pca)
 
        PASTE=1

        gmm_predicted=gmm.predict_proba(X_train_pca)
        print gmm_predicted


    X_train_dict = getFeatures(inputd,cluster=cluster,noGrep=args.noGrep,ngram=args.ngram,gmm=gmm_predicted)
    print >>sys.stderr, u"Features: {}".format(X_train_dict[-2])
    

    data['data'] = vectorizer.fit_transform(X_train_dict)
    #print vectorizer.vocabulary_
        

    if args.cv:
        num_inst=len(labels)
        train_cv={}
        cross=cross_validation.KFold(len(labels),n_folds=10) 
        acc=[]
        for train_index, test_index in cross:
            if args.debug:
                print("TRAIN:", len(train_index), "TEST:", len(test_index))
            X=data['data']
            y=data['target']
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model= Ridge(alpha=args.alpha)
            model.fit(X_train,y_train)
    
            y_pred= model.predict(X_test)
            assert(len(y_pred)==len(test_index))
            tids=tweetids[test_index]
            for twid,pred in zip(tids,y_pred):
                train_cv[twid] =  pred
            
            y_pred_official=[math.floor(x + 0.5) for x in y_pred]
            y_gold_official=[math.floor(x + 0.5) for x in y_test]
            acc.append(cosine_similarity(y_gold_official,y_pred_official)[0][0])
            if args.debug:
                evaluate(y_test,y_pred,plot=args.plot)
            
        print >>sys.stderr, "Cosine of 10-folds:", acc
        print >>sys.stderr, "Macro average:", np.mean(np.array(acc)), np.std(np.array(acc))
        
        if args.cvp:
            for twid in tweetids:
                print "{}\t{}".format(twid,train_cv[twid])

    ### test data
    tweetids,testd,y_test=read_data_file(args.test,args.removeHashTags)
    if args.gmm:
        ## get features for gmm on test data
        X_test_dict = getFeatures(testd,cluster=cluster,noGrep=args.noGrep,ngram=args.ngram) #without gmm
        X_test=vectorizernogmm.transform(X_test_dict)
        X_test_pca = pca.transform(X_test)
        gmm_predicted=gmm.predict_proba(X_test_pca)
        print gmm_predicted

    X_test_dict = getFeatures(testd,cluster=cluster,noGrep=args.noGrep,ngram=args.ngram,gmm=gmm_predicted)
    X_test=vectorizer.transform(X_test_dict)


    
    
    print >>sys.stderr, "Train on whole, eval on trial"
    if args.compare:
        print "LinearReg"
        model=LinearRegression()
        model.fit(data['data'],data['target'])
        y_pred= model.predict(X_test)
        evaluate(y_test,y_pred)

    print >>sys.stderr,  "Ridge"
    #model=RidgeCV([0.00000001,0.001,0.01,0.0001,0.1,1.0,1.5,2,10]) #alpha=args.alpha)
    model=Ridge(alpha=args.alpha)
    #model=Lasso(alpha=args.alpha)
    #model=ElasticNet(alpha=args.alpha, l1_ratio=0.7)
    model.fit(data['data'],data['target'])
    y_pred= model.predict(X_test)
    y_pred=y_pred    
    evaluate(y_test,y_pred,plot=args.plot)
    show_most_informative_features(vectorizer,model)

    #from sklearn.grid_search import GridSearchCV
    
    alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
    # create and fit a ridge regression model, testing each alpha
    #model = Ridge()
    #grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
    #grid.fit(X_all,y_all)
    #print(grid)
    # summarize the results of the grid search
    #print(grid.best_score_)
    #print(grid.best_estimator_.alpha)

    print >>sys.stderr,"coef_:",model.coef_
    print >>sys.stderr,model

    if args.out:
        for i,j in zip(y_test,y_pred):
            print i,j

    if args.pred and not args.out:
        for i,val in enumerate(y_pred):
            print u"{}\t{}\t{}".format(tweetids[i],val,inputd[i])

    ## use +/- classifier label and stacking
#     echo "TO FINISH..."
#     cross=cross_validation.KFold(len(labels),n_folds=10)
#     acc=[]
#     plusminuslabels=np.zeros(len(data['target']))*-1
#     for train_index, test_index in cross:
#         X=data['data']
#         y=data['target']
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#         y_train=[0 if x < 0 else 1 for x in y_train]
#         y_test=[0 if x < 0 else 1 for x in y_test]
#         print >>sys.stderr, Counter(y_train)
#         classifier= LogisticRegression(class_weight={0: args.classweight})
#         #classifier= LogisticRegression()
#         classifier.fit(X_train,y_train)
#         class_pred=classifier.predict(X_test)
#         print >>sys.stderr, "accuracy for +/-: {}".format(accuracy_score(y_test,class_pred))
#         plusminuslabels[test_index] = class_pred
        
#     print plusminuslabels
#     print len(plusminuslabels)
#     y_train=[0 if x < 0 else 1 for x in data['target']]
#     print Counter(plusminuslabels)
#     print >>sys.stderr, "accuracy for +/-: {}".format(accuracy_score(y_train,plusminuslabels))
    
    ## create pandas object
    #dat=pd.DataFrame(data=plusminuslabels,columns=["pm"])
    #plusminusdict=dat.T.to_dict().values() #transpose! (to not have indices as keys)
    #print plusminusdict
    #plusmins=DictVectorizer(plusminusdict)

    ##### add plusminus as additional feature
    #vectorizer=FeatureUnion([("pm",plusmins),("w",vectorizerWord)])
    #data['data'] = vectorizer.fit_transform(inputd)    




def evaluate(y_gold,y_pred,plot=False):
    print >>sys.stderr, "mean absolute error", mean_absolute_error(y_gold, y_pred)
    print >>sys.stderr, "MSE", mean_squared_error(y_gold, y_pred)
    print >>sys.stderr, "R2", r2_score(y_gold, y_pred)
    r_row, p_value = pearsonr(y_gold,y_pred)
    print >>sys.stderr, "Pearsonr {}".format(r_row)
    print >>sys.stderr, "Cosine", cosine_similarity(y_gold,y_pred)[0][0]
    y_pred_official=[math.floor(x + 0.5) for x in y_pred]
    y_gold_official=[math.floor(x + 0.5) for x in y_gold]
    print >>sys.stderr, "Cosine official (rounded)", cosine_similarity(y_gold_official,y_pred_official)[0][0]
    if plot:
        plt.subplot(2, 1, 1)
        plt.scatter(y_gold,y_pred)
        plt.subplot(2, 1, 2)
        plt.scatter(y_gold_official,y_pred_official)
        plt.show()
        

    

def read_data_file(datafile,removeHashTags):
    inputd=[]
    labels=[]
    tweetids=[]
    for line in codecs.open(datafile,encoding="utf-8"):            
        id,label,text=line.strip().split("\t",2)
        tweet = preproc.replace_user_tags(text)

        if removeHashTags == True:
            tweet = removeHashTags(tweet)
        
        labels.append(float(label))
        inputd.append(tweet)
        tweetids.append(id)
    assert(len(inputd)==len(labels))
    return tweetids,inputd, labels


def show_most_informative_features(vectorizer, clf, n=10):
    feature_names = vectorizer.get_feature_names()
    
    coefs_with_fns = sorted(zip(clf.coef_, feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print >>sys.stderr,"\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)


# This list of English stop words is taken from the "Glasgow Information
# Retrieval Group". The original list can be found at
# http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words
ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fify", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])
    

if __name__=="__main__":
    main()
