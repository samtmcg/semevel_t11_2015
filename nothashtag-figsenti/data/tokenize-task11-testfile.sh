#!/bin/bash

awk -F'\t' '{print $2}' organizers/testDataRaw.dat > /tmp/$$tmp
sh twokenize.sh < /tmp/$$tmp > /tmp/$$tmp.twok
awk -F'\t' '{printf "%s\t%s\n", $1,"0.0"}' organizers/testDataRaw.dat > /tmp/$$tmp.lab
paste /tmp/$$tmp.lab /tmp/$$tmp.twok > twokenized/test.dat

