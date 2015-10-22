#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import codecs
import sys
import numpy as np
import re
import preproc
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

from sklearn import cross_validation
from sklearn.pipeline import FeatureUnion
from collections import Counter
import pandas as pd
from sklearn import decomposition
from sklearn.mixture import GMM
from sklearn import svm
import pylab as plt

def main():

    parser = argparse.ArgumentParser(description="Ridge regression model")
    parser.add_argument('train', help="train data")
    parser.add_argument('test', help="test data")
    parser.add_argument('--seed', help="random seed",type=int,default=987654321)
    parser.add_argument('--debug', help="debug", action='store_true', default=False)
    parser.add_argument('--removeHashTags', help="removeHashTags", action='store_true', default=False)
    parser.add_argument('--alpha', help="alpha parameter", type=float, default=1.0)
    parser.add_argument('--classweight', help="class weight for +/- classifier", type=float, default=22)
    parser.add_argument('--out', help="output gold and predictions", action='store_true', default=False)
    parser.add_argument('--compare', help="compare to linReg", action='store_true', default=False)
    parser.add_argument('--pred', help="output predictions", action='store_true', default=False)
    parser.add_argument('-c','--components', help="num PCA components", type=int, default=13)
    parser.add_argument('-g','--gmmcomponents', help="num GMM components", type=int, default=2)
    parser.add_argument('--plot', help="plot predictions", action='store_true', default=False)
    parser.add_argument('--cv', help="run cross-validation", action='store_true', default=False)
    parser.add_argument('--cvp', help="output Lcv scores", action='store_true', default=False)

    args = parser.parse_args()

    
    ####### load data
    tweetidstrain,inputd,labels=read_data_file(args.train,args.removeHashTags)
    print >>sys.stderr, "{} instances loaded.".format(len(inputd))
    print >>sys.stderr, u"Instance: {} {}".format(labels[-2],inputd[-2])

    tweetidstrain=np.array(tweetidstrain)
    np.random.seed(args.seed)
    
    data ={}

    print >>sys.stderr, len(labels)
    # target labels

    data['target'] = np.array(labels)

    # now vectorize data and save as 'data' (better: DictVectorizer!!)
    #vectorizer = CountVectorizer(analyzer="word",binary=False)
    vectorizerWord = CountVectorizer(analyzer="word",binary=True,ngram_range=(1,3))


    vectorizer=vectorizerWord
    data['data'] = vectorizer.fit_transform(inputd)


    pca = decomposition.RandomizedPCA(n_components=args.components)
    #pca = decomposition.TruncatedSVD(n_components=args.components)
    X_train=data['data']

    X_train_pca = pca.fit_transform(X_train)

    gmm = GMM(n_components = args.gmmcomponents,
              covariance_type = "full",
              min_covar = 0.01)
    y_train=data['target']


    gmm.fit(X_train_pca)
 
    PASTE=1
    CAT=0

    S1=pd.DataFrame(data=gmm.predict_proba(X_train_pca))


    #meta = Ridge() #svm.SVC()#C=args.svmC,gamma=args.svmGamma)
    meta=Ridge()
    #meta=svm.SVC()
    meta.fit(S1,y_train)

    #get test data
    tweetids,testd,y_test=read_data_file(args.test,args.removeHashTags)
    X_test=vectorizer.transform(testd)

    tweetids=np.array(tweetids)
    X_test_pca=pca.transform(X_test)

    T1=pd.DataFrame(data=gmm.predict_proba(X_test_pca))

    y_pred_gmm= meta.predict(T1)
    #print Counter(y_pred_gmm)

    evaluate(y_test,y_pred_gmm)


    if args.pred and not args.out:
        for i,val in enumerate(y_pred_gmm):
            print u"{}\t{}\t{}".format(tweetids[i],val,inputd[i])


    if args.cv:
        num_inst=len(labels)
        print >>sys.stderr, num_inst
        train_cv={}
        cross=cross_validation.KFold(len(labels),n_folds=10)
        acc=[]
        for train_index, test_index in cross:
            if args.debug:
                print("TRAIN:", len(train_index), "TEST:", len(test_index))
                
            X=S1.as_matrix()
            y=data['target']
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print >>sys.stderr, len(X_train), len(y_test)       , len(tweetids)

            model= Ridge(alpha=args.alpha)
            model.fit(X_train,y_train)

            y_pred= model.predict(X_test)
            assert(len(y_pred)==len(test_index))
            tids=tweetidstrain[test_index]
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
            for twid in tweetidstrain:
                print "{}\t{}".format(twid,train_cv[twid])


    if args.plot:
    #fig, axes = plot.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
        plt.subplot(3, 1, 1)
        plt.scatter(y_test,y_pred_gmm)
    #plt.show()

        plt.subplot(3, 1, 2)
        plt.hist(y_pred_gmm)
        plt.title("pred")

        plt.subplot(3, 1, 3)
        plt.hist(y_test)
        plt.title("gold")
        plt.show()



def evaluate(y_gold,y_pred):
    print >>sys.stderr, "mean absolute error", mean_absolute_error(y_gold, y_pred)
    print >>sys.stderr, "MSE", mean_squared_error(y_gold, y_pred)
    print >>sys.stderr, "R2", r2_score(y_gold, y_pred)
    r_row, p_value = pearsonr(y_gold,y_pred)
    print >>sys.stderr, "Pearsonr {}".format(r_row)
    print >>sys.stderr, "Cosine", cosine_similarity(y_gold,y_pred)[0][0]

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



    

if __name__=="__main__":
    main()
