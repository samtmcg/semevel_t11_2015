#!/bin/bash

awk -F'\t' '{print $3}' $1 > /tmp/$$tmp
sh twokenize.sh < /tmp/$$tmp > /tmp/$$tmp.twok
awk -F'\t' '{printf "%s\t%s\n", $1,$2}' $1 > /tmp/$$tmp.lab
paste /tmp/$$tmp.lab /tmp/$$tmp.twok > $1.tok

