#!/usr/bin/bash
# run their official scorer on our output files
#
# sh run-semevalscorer.sh GOLD PRED
if [ -z "$1" ] || [ -z "$2" ] 
then
    echo "Usage: sh run-semevalscorer.sh GOLD PRED"
    exit
fi
awk -F'\t' '{printf "%s\t%s\n",$1,$2}' $1 > $$tmp1
awk -F'\t' '{printf "%s\t%s\n",$1,$2}' $2 > $$tmp2
python semevalscorer.py $$tmp1 $$tmp2 
rm $$tmp1 $$tmp2

