#!/bin/bash
fileprefix=$1
for file1 in `ls ../data/$fileprefix*`
do
    outstring=`basename $file1`
        for file2 in `ls ../runs/$fileprefix*`
        do
            perf=`python correlator.py $file1 $file2`
            outstring=$outstring","$perf
        done
    echo $outstring
done