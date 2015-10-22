#!/bin/bash
#run ridge
PYTHONIOENCODING=utf-8 python semeval_ridge_regression.py ../data/twokenized/train.dat ../data/twokenized/trial.dat --labelsTest ../data/twokenized/trial.labels --labelsTrain ../data/twokenized/train.labels --pred > ../data/zvals/trial.ridge.1-3.nostop.pluslabel.plusSimple