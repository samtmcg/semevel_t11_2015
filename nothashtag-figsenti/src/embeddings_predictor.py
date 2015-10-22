# -*- coding: utf-8 -*-
import numpy as np
import sys
import argparse
import codecs
import numpy as np
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression, LinearRegression, BayesianRidge
from sklearn import tree
from sklearn import cross_validation
from sklearn.metrics.pairwise import cosine_similarity

from sklearn import preprocessing
from sklearn import metrics
import re
dimensions = 100



def get_embedding(word,embeddings):
    embed = []
    if word in embeddings:
        embed=embeddings[word]
    else:
        embed = [0.0] * dimensions
    return embed

def valid(word,embeddings,stoplist):
    if word in stoplist:
        return False
    if word not in embeddings:
        return False
    if re.search("[a-z]",word) is None: #purge numbers
        return False
    return True


def filteruser(word):
    if word.startswith("@"):
        return "@user"
    else:
        return word


def embedfeats(sentence,embeddings,stoplist):
    embeds = []
    sentence = [w.lower() for w in sentence]
    sentence = [filteruser(w) for w in sentence if valid(w,embeddings,stoplist)]


    for word in sentence:
        if not word in stoplist:
            embeds.append(get_embedding(word,embeddings))

    if len(embeds) == 0:
        embeds = [get_embedding(".",embeddings)]

    #print str(len(embeds))+"\t"+" ".join(sentence)


    cos_bm = 0.0
    cos_be = 0.0
    cos_me = 0.0
    if len(embeds) > 2:
        (beginning, middle, end)= np.array_split(embeds,3)
        b = max_min_embed(beginning,"avg")
        m = max_min_embed(middle,"avg")
        e = max_min_embed(end,"avg")
        cos_bm = metrics.pairwise.cosine_similarity(b,m)[0][0]
        cos_be = metrics.pairwise.cosine_similarity(b,e)[0][0]
        cos_me = metrics.pairwise.cosine_similarity(m,e)[0][0]


    #outfeats = [cos_be,cos_bm,cos_me]
    outfeats = []

    #outfeats.extend(max_min_embed(embeds,"max"))
    #outfeats.extend(max_min_embed(embeds,"min"))
    outfeats.extend(max_min_embed(embeds,"avg"))
    return outfeats


def max_min_embed(embedingslist, operation):
    maxvect = []
    for d in range(dimensions):
        acc = []
        for vector in range(len(embedingslist)):
            acc.append(embedingslist[vector][d])
        if operation == "max":
            maxvect.append(max(acc))
        elif operation == "min":
            maxvect.append(min(acc))
        elif operation == "avg":
            maxvect.append(sum(acc)/len(acc))

        #print >> sys.stderr,maxvect
    return maxvect





def roundup(value):
    if value < 0:
        return int(value-0.5)
    return int(value+0.5)


def main():
    parser = argparse.ArgumentParser(description="""Creates embeddings predictions.""")
    parser.add_argument('--train')
    parser.add_argument('--test')
    parser.add_argument('--embeddings')
    parser.add_argument('--cv',default=False)


    args = parser.parse_args()

    stoplist = stopwords.words("english")
    stoplist.extend("it's 've 's i'm he's she's you're we're they're i'll you'll he'll ".split(" "))


    embeddings={}
    for line in codecs.open(args.embeddings,encoding="utf-8").readlines():
        line = line.strip()
        if line:
            a= line.split(" ")
            embeddings[a[0]] = np.array([float(v) for v in a[1:]]) #cast to float, otherwise we cannot operate

    train_indices = []
    test_indices = []
    train_scores = []
    train_features = []
    test_features = []


    # if args.learner == "logisticregression":
    #     learner= LogisticRegression()
    #     learner_type = "classification"
    # elif args.learner == "decisiontreeclassification":
    #     learner = tree.DecisionTreeClassifier()
    #     learner_type = "classification"
    # elif args.learner == "decisiontreeregression":
    #     learner = tree.DecisionTreeRegressor()
    #     learner_type = "regression"
    # elif args.learner == "bayesianridge":
    #     learner = BayesianRidge()
    #     learner_type = "regression"
    # else:
    learner = BayesianRidge()
    learner_type = "regression"

    le = preprocessing.LabelEncoder()


    for line in open(args.train).readlines():
        (index, score, tweet) = line.strip().split("\t")
        train_indices.append(index)
        train_scores.append(float(score))
        tweet = tweet.split(" ")
        train_features.append(embedfeats(tweet,embeddings,stoplist))


    train_indices = np.array(train_indices)
    train_scores = np.array(train_scores)
    train_features = np.array(train_features)

    train_scores_int = [roundup(v) for v in train_scores]
    le.fit(train_scores_int)

    train_scores_int_transformed = le.transform(train_scores_int)


    if args.cv:
        train_cv={}
        cross=cross_validation.KFold(len(train_scores),n_folds=10)
        acc=[]
        for train_index, test_index in cross:
            #if args.debug:
            #    print("TRAIN:", len(train_index), "TEST:", len(test_index))
            X=train_features
            y=train_scores
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]


            learner.fit(X_train,y_train)

            y_pred= learner.predict(X_test)
            assert(len(y_pred)==len(test_index))
            tids=train_indices[test_index]
            for twid,pred in zip(tids,y_pred):
                train_cv[twid] =  pred

            acc.append(cosine_similarity(y_test,y_pred)[0][0])

        print >>sys.stderr, "Cosine of 10-folds:", acc
        print >>sys.stderr, "Macro average:", np.mean(np.array(acc)), np.std(np.array(acc))

        for twid in train_indices:
            print "{}\t{}".format(twid,train_cv[twid])
    else:

        for line in open(args.test).readlines():
            (index, score, tweet) = line.strip().split("\t")
            test_indices.append(index)
            #scores.append(score)
            tweet = tweet.split(" ")
            test_features.append(embedfeats(tweet,embeddings,stoplist))


        #print  np.array(train_features).shape
        # when features are generated, train and test

        if learner_type == "regression":
            learner.fit(train_features,train_scores)
        else:
                learner.fit(train_features,train_scores_int_transformed)

        predicted_scores= learner.predict(test_features)
        if learner_type != "regression":
            predicted_scores = le.inverse_transform(predicted_scores)
        for index, score in zip(test_indices,predicted_scores):
            print index+"\t"+str(score)
        #print >>sys.stderr, "cosine similarity:", cosine_similarity(predicted_scores,y_pred)[0][0]

main()