TRAINFINAL=../data/twokenized/train+trial.dat
TESTFINAL=../data/twokenized/test.dat

TRAIN=../data/twokenized/train.dat
TEST=../data/twokenized/trial.dat
#### final system runs as input for the ensemble

#constant baseline: can directly run on train itself since constant
##python create_constant_baseline.py -t average ../data/twokenized/train.dat ../data/twokenized/trial.dat > ../data/zvals/dev/trial.constant-baseline-average
##python create_constant_baseline.py -t average ../data/twokenized/train.dat ../data/twokenized/train.dat > ../data/zvals/dev/train.constant-baseline-average

### final
##python create_constant_baseline.py -t average ../data/twokenized/train+trial.dat ../data/twokenized/test.dat > ../data/zvals/final/test.constant-baseline-average
##python create_constant_baseline.py -t average ../data/twokenized/train+trial.dat ../data/twokenized/train+trial.dat > ../data/zvals/final/train.constant-baseline-average


### grep labeler
##PYTHONIOENCODING=utf-8 python label.py $TRAIN -t $TEST -q > ../data/zvals/dev/trial.greplabel 
##PYTHONIOENCODING=utf-8 python label.py $TRAIN -p > ../data/zvals/dev/train.greplabel

##PYTHONIOENCODING=utf-8 python label.py $TRAINFINAL -t $TESTFINAL -q > ../data/zvals/final/trial.greplabel 
##PYTHONIOENCODING=utf-8 python label.py $TRAINFINAL -p > ../data/zvals/final/train.greplabel

##PYTHONIOENCODING=utf-8 python label.py -z $TRAIN -t $TEST -q > ../data/zvals/dev/trial.greplabel.zero
##PYTHONIOENCODING=utf-8 python label.py -z $TRAIN -p > ../data/zvals/dev/train.greplabel.zero

##PYTHONIOENCODING=utf-8 python label.py -z $TRAINFINAL -t $TESTFINAL -q > ../data/zvals/final/trial.greplabel.zero 
##PYTHONIOENCODING=utf-8 python label.py -z $TRAINFINAL -p > ../data/zvals/final/train.greplabel.zero
#exit

##### ridge baseline
# PYTHONIOENCODING=utf-8 python semeval_ridge_regression.py ../data/twokenized/train.dat ../data/twokenized/trial.dat --cv --cvp > ../data/zvals/dev/train.ridge.baseline

# PYTHONIOENCODING=utf-8 python semeval_ridge_regression.py ../data/twokenized/train.dat ../data/twokenized/trial.dat --pred > ../data/zvals/dev/trial.ridge.baseline

# PYTHONIOENCODING=utf-8 python semeval_ridge_regression.py $TRAINFINAL $TESTFINAL --cv --cvp > ../data/zvals/final/train.ridge.baseline

# PYTHONIOENCODING=utf-8 python semeval_ridge_regression.py $TRAINFINAL $TESTFINAL --pred > ../data/zvals/final/test.ridge.baseline

# PYTHONIOENCODING=utf-8 python semeval_ridge_regression.py --noGrep ../data/twokenized/train.dat ../data/twokenized/trial.dat --cv --cvp > ../data/zvals/dev/train.ridge.baseline.noGrep

# PYTHONIOENCODING=utf-8 python semeval_ridge_regression.py --noGrep ../data/twokenized/train.dat ../data/twokenized/trial.dat --pred > ../data/zvals/dev/trial.ridge.baseline.noGrep

# PYTHONIOENCODING=utf-8 python semeval_ridge_regression.py --noGrep $TRAINFINAL $TESTFINAL --cv --cvp > ../data/zvals/final/train.ridge.baseline.noGrep

# PYTHONIOENCODING=utf-8 python semeval_ridge_regression.py --noGrep $TRAINFINAL $TESTFINAL --pred > ../data/zvals/final/test.ridge.baseline.noGrep

### 1,2,3 gram ridge

for ngram in 1 1-2 1-2-3 1-2-3-4 1-2 1-3 1-4 1-5 1-2-4 1-2-5
do
    PYTHONIOENCODING=utf-8 python semeval_ridge_regression.py --ngram $ngram $TRAIN $TEST --cv --cvp > ../data/zvals/dev/train.ridge.baseline.ngram-$ngram
    PYTHONIOENCODING=utf-8 python semeval_ridge_regression.py --ngram $ngram $TRAIN $TEST --pred > ../data/zvals/dev/trial.ridge.baseline.ngram-$ngram
    PYTHONIOENCODING=utf-8 python semeval_ridge_regression.py --ngram $ngram $TRAINFINAL $TESTFINAL --pred > ../data/zvals/final/test.ridge.baseline.ngram-$ngram
    PYTHONIOENCODING=utf-8 python semeval_ridge_regression.py --ngram $ngram $TRAINFINAL $TESTFINAL --cv --cvp > ../data/zvals/final/train.ridge.baseline.ngram-$ngram
done

exit

####gmm 

#PYTHONIOENCODING=utf-8 python gmm.py ../data/twokenized/train.dat ../data/twokenized/trial.dat -c 100 -g 12 --pred > ../data/zvals/dev/trial.gmm.c100.g12
#PYTHONIOENCODING=utf-8 python gmm.py ../data/twokenized/train.dat ../data/twokenized/trial.dat -c 100 -g 12 --cv --cvp > ../data/zvals/dev/train.gmm.c100.g12

#PYTHONIOENCODING=utf-8 python gmm.py $TRAINFINAL $TESTFINAL -c 100 -g 12 --pred > ../data/zvals/final/test.gmm.c100.g12
#PYTHONIOENCODING=utf-8 python gmm.py $TRAINFINAL $TESTFINAL -c 100 -g 12 --cv --cvp > ../data/zvals/final/train.gmm.c100.g12

####

## 1,2,3 gramlogistic regressor

PYTHONIOENCODING=utf-8 python ngrams_logistic_regression_baseline.py ../data/twokenized/train.dat ../data/twokenized/trial.dat -t count -rht False -rsw True -ngrams '1 3' -cv True > ../data/zvals/dev/train.onetwothree_logistic

PYTHONIOENCODING=utf-8 python ngrams_logistic_regression_baseline.py ../data/twokenized/train.dat ../data/twokenized/trial.dat -t count -rht False -rsw True -ngrams '1 3'  > ../data/zvals/dev/trial.onetwothree_logistic

PYTHONIOENCODING=utf-8 python ngrams_logistic_regression_baseline.py $TRAINFINAL $TESTFINAL -t count -rht False -rsw True -ngrams '1 3' -cv True  > ../data/zvals/final/train.onetwothree_logistic
PYTHONIOENCODING=utf-8 python ngrams_logistic_regression_baseline.py $TRAINFINAL $TESTFINAL -t count -rht False -rsw True -ngrams '1 3' -cv False  > ../data/zvals/final/test.onetwothree_logistic

# 3 gram logistic regressor

PYTHONIOENCODING=utf-8 python ngrams_logistic_regression_baseline.py ../data/twokenized/train.dat ../data/twokenized/trial.dat -t count -rht False -rsw True -ngrams '3 3' -cv True > ../data/zvals/dev/train.three_logistic

PYTHONIOENCODING=utf-8 python ngrams_logistic_regression_baseline.py ../data/twokenized/train.dat ../data/twokenized/trial.dat -t count -rht False -rsw True -ngrams '3 3'  > ../data/zvals/dev/trial.three_logistic

PYTHONIOENCODING=utf-8 python ngrams_logistic_regression_baseline.py $TRAINFINAL $TESTFINAL -t count -rht False -rsw True -ngrams '3 3' -cv True  > ../data/zvals/final/train.three_logistic
PYTHONIOENCODING=utf-8 python ngrams_logistic_regression_baseline.py $TRAINFINAL $TESTFINAL -t count -rht False -rsw True -ngrams '3 3' -cv False  > ../data/zvals/final/test.three_logistic

# 2 gram logistic regressor

PYTHONIOENCODING=utf-8 python ngrams_logistic_regression_baseline.py ../data/twokenized/train.dat ../data/twokenized/trial.dat -t count -rht False -rsw True -ngrams '2 2' -cv True > ../data/zvals/dev/train.two_logistic

PYTHONIOENCODING=utf-8 python ngrams_logistic_regression_baseline.py ../data/twokenized/train.dat ../data/twokenized/trial.dat -t count -rht False -rsw True -ngrams '2 2'  > ../data/zvals/dev/trial.two_logistic

PYTHONIOENCODING=utf-8 python ngrams_logistic_regression_baseline.py $TRAINFINAL $TESTFINAL -t count -rht False -rsw True -ngrams '2 2' -cv True  > ../data/zvals/final/train.two_logistic
PYTHONIOENCODING=utf-8 python ngrams_logistic_regression_baseline.py $TRAINFINAL $TESTFINAL -t count -rht False -rsw True -ngrams '2 2' -cv False  > ../data/zvals/final/test.two_logistic

