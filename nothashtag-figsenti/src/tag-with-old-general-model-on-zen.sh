#!/bin/bash
### input: figsenti file with id\tscore\ttext

cat $1 | awk -F'\t' '{print $3}' > /tmp/$$tmp
DIR=/home/dirkh/lowlands/sentiment-analysis/english/hector
cd $DIR
sh tag.sh /tmp/$$tmp 
rm /tmp/$$tmp